import argparse
import os
import sys
sys.path.append('./utils')
from utils import *
import numpy as np
import uuid
import time
import requests
import base64
from openai import OpenAI
import openai
from mask_utils import *
from tqdm import trange
from augmentations import *
import pickle
import spacy
from PIL import Image
import shutil

import json
from tqdm import tqdm
openai_api_key = "EMPTY"
openai_api_base = 'http://0.0.0.0:23333/v1'
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
def get_method(method_name):
    try:
        method = img_aug_dict[method_name]
        method = img_aug_dict[method_name]
    except:
        print('Check your method!!')
        os._exit(0)
    return method
all_req_time = 0
def get_request(model, query, img, url):
    t_1 = time.time()
    if model == 'minigpt4':
        while True:
            try:
                url = url
                input_data = {
                    "inputs": [
                        {
                            "name": "text_input",
                            "shape": [1],
                            "datatype": "BYTES",
                            "data": [query]
                        },

                        {
                            "name": "temperature",
                            "shape": [1],
                            "datatype": "FP32",
                            "data": [0.01]
                        },

                        {
                            "name": "imgbase64",
                            "shape": [1],
                            "datatype": "BYTES",
                            "data": [img]
                        }
                    ],
                    "outputs": [
                        {
                            "name": "OUTPUT0"
                        }
                    ]
                }
                results = requests.post(url, timeout=300, json=input_data)
                answer = json.loads(results.text)['outputs'][0]['data'][0]

            except requests.exceptions.ConnectionError:
                print('ConnectionError -- please wait 3 seconds')
                time.sleep(3)
                continue
            except requests.exceptions.ChunkedEncodingError:
                print('ChunkedEncodingError -- please wait 3 seconds')
                time.sleep(3)
                continue
            except:
                print('Unfortunitely -- An Unknow Error Happened, Please wait 30 seconds')
                time.sleep(30)
                continue
            break
    elif model == 'llava':
        while True:
            try:
                chat_response = client.chat.completions.create(
                    model="/llm/llava-llama-2-13b-chat-lightning-preview",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img}",
                                },
                            },
                        ],
                    }],
                )
                answer = chat_response.choices[0].message.content
            except:
                print('Unfortunitely -- An Unknow Error Happened, Please wait 30 seconds')
                time.sleep(30)
                continue
            break
    t_2 = time.time()
    req_time = (t_2 - t_1)
    return answer, req_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Text Experiment')
    parser.add_argument('--mutator', default='PL', type=str, help='Horizontal Flip(HF),Vertical Flip(VF),Random Rotation(RR),Crop and Resize(CR),Random Mask(RM),Random Solarization(RS),Random Grayscale(GR),Gaussian Blur(BL), Colorjitter(CJ), Random Posterization(RP) Policy(PL)')
    parser.add_argument('--variant_save_dir', default='demo_case/variant', type=str, help='dir to save the modify results')
    parser.add_argument('--response_save_dir', default='demo_case/response', type=str, help='dir to save the modify results')
    parser.add_argument('--number', default='8', type=str, help='number of generated variants')
    parser.add_argument('--threshold', default=0.025, type=str, help='Threshold of divergence')

    parser.add_argument("--list_path", type=str, default='../../llava_attack_success_test_hh')
    parser.add_argument("--raw_image_fold", type=str, default="../../eff_imgs")
    parser.add_argument("--model", type=str, default="minigpt4")
    parser.add_argument("--url", type=str, default='http://minigpt4.nextcenter.net:58881/v2/models/minigpt/infer')
    parser.add_argument('--type', default='all', type=str, help='adversarial or benign or all')

    args = parser.parse_args()

    attacked_image_fold = '../../eff_imgs'

    start_time = time.time()

    variant_save_dir = 'demo_case/' + args.list_path.split("../../")[1] + 'var'
    response_save_dir = 'demo_case/' + args.list_path.split("../../")[1] + 'res'
    number=int(args.number)

    # Step1: mask input

    print(args)
    as_qa = []
    with open(os.path.join(args.list_path, 'success_qa.json'), 'r') as f:
        as_qa = json.load(f)


    attack_count_succesful = 0
    raw_count_succesful = 0

    # vis_processor, chat, model = initialize_model()  # initialize minigpt-4 model.refer to https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/minigpt_inference.py
    as_qa = as_qa[:10]
    if args.type == 'all' or 'adversarial':
        for item in tqdm(as_qa):

            # Step1: mask input
            img_path = os.path.join(attacked_image_fold, item["img"])
            query = item["query"]
            if not os.path.exists(variant_save_dir):
                os.makedirs(variant_save_dir)
            target_dir = variant_save_dir

            for i in range(number):
                tmp_method = get_method(args.mutator)
                pil_img = Image.open(img_path)
                new_image = tmp_method(img=pil_img)

                # save image
                if '.bmp' in img_path:
                    target_path = os.path.join(target_dir, str(i)+f'-{args.mutator}.bmp')
                else:
                    target_path = os.path.join(target_dir, str(i)+f'-{args.mutator}.jpg')
                if len(os.listdir(target_dir)) >= number+1:
                    break # early stop
                # cv2.imwrite()
                # creating a image object (main image)
                new_image.save(target_path)


            # Step2: query_model
            variant_list, name_list = load_mask_dir(target_dir)
            question = ''.join(query)
            new_save_dir = response_save_dir
            if not os.path.exists(new_save_dir):
                os.makedirs(new_save_dir)
            for j in range(len(variant_list)):
                img_prompt_path = variant_list[j]
                prompts_eval = [question, img_prompt_path]
                # read_file_in_line(mask_file_list[i])
                save_name = name_list[j].split('.')[0]
                existing_response = [i for i in os.listdir(new_save_dir) if'.png' not in i]
                if len(existing_response) >= number:
                    continue
                new_save_path = os.path.join(new_save_dir, save_name)
                if not os.path.exists(new_save_path):

                    #result = model_inference(vis_processor, chat, model, prompts_eval)
                    image_file = open(img_prompt_path, 'rb')
                    img = base64.b64encode(image_file.read()).decode('utf-8')
                    result, req_time = get_request(args.model, question, img, args.url)
                    all_req_time += req_time
                    f = open(new_save_path, 'w')
                    f.writelines(result)
                    f.close()

            # Step3: divergence & detect
            diver_save_path=os.path.join(response_save_dir,f'diver_result-{args.number}.pkl')
            metric = spacy.load("en_core_web_md")
            avail_dir = response_save_dir
            check_list = os.listdir(avail_dir)
            check_list = [os.path.join(avail_dir, check) for check in check_list]
            output_list = read_file_list(check_list)
            max_div, jailbreak_keywords = update_divergence(output_list, 1, avail_dir, select_number=number, metric=metric, top_string=100)
            detection_result = detect_attack(max_div, jailbreak_keywords, args.threshold)
            if detection_result:
                attack_count_succesful+=1
            shutil.rmtree(variant_save_dir)
            shutil.rmtree(response_save_dir)

    if args.type == 'all' or 'benign':
        for item in tqdm(as_qa):

            img_path = os.path.join(args.raw_image_fold, item["img"].replace('bmp', 'jpg'))
            query = item["query"]
            if not os.path.exists(variant_save_dir):
                os.makedirs(variant_save_dir)
            target_dir = variant_save_dir
            for i in range(number):
                tmp_method = get_method(args.mutator)
                pil_img = Image.open(img_path)
                new_image = tmp_method(img=pil_img)

                # save image
                if '.bmp' in img_path:
                    target_path = os.path.join(target_dir, str(i)+f'-{args.mutator}.bmp')
                else:
                    target_path = os.path.join(target_dir, str(i)+f'-{args.mutator}.jpg')
                if len(os.listdir(target_dir)) >= number+1:
                    break # early stop
                # cv2.imwrite()
                # creating a image object (main image)

                new_image.save(target_path)


            # Step2: query_model
            variant_list, name_list = load_mask_dir(target_dir)
            question = ''.join(query)
            new_save_dir = response_save_dir
            if not os.path.exists(new_save_dir):
                os.makedirs(new_save_dir)
            for j in range(len(variant_list)):
                img_prompt_path = variant_list[j]
                prompts_eval = [question, img_prompt_path]
                # read_file_in_line(mask_file_list[i])
                save_name = name_list[j].split('.')[0]
                existing_response = [i for i in os.listdir(new_save_dir) if'.png' not in i]
                if len(existing_response) >= number:
                    continue
                new_save_path = os.path.join(new_save_dir, save_name)
                if not os.path.exists(new_save_path):
                    image_file = open(img_prompt_path, 'rb')
                    img = base64.b64encode(image_file.read()).decode('utf-8')
                    result, req_time = get_request(args.model, question, img, args.url)
                    all_req_time += req_time
                    f = open(new_save_path, 'w')
                    f.writelines(result)
                    f.close()

            # Step3: divergence & detect
            diver_save_path=os.path.join(response_save_dir,f'diver_result-{args.number}.pkl')
            metric = spacy.load("en_core_web_md")
            avail_dir = response_save_dir
            check_list = os.listdir(avail_dir)
            check_list = [os.path.join(avail_dir, check) for check in check_list]
            output_list = read_file_list(check_list)
            max_div, jailbreak_keywords = update_divergence(output_list, 1, avail_dir, select_number=number, metric=metric, top_string=100)
            detection_result = detect_attack(max_div, jailbreak_keywords, args.threshold)
            if not detection_result:
                raw_count_succesful += 1
            shutil.rmtree(variant_save_dir)
            shutil.rmtree(response_save_dir)
    end_time = time.time()
    print(f"时间: {end_time - start_time:.6f} 秒")
    print(f"request时间：{all_req_time}")
    # print(f"acc: {(attack_count_succesful + raw_count_succesful) / (1000) * 100:.2f}%")