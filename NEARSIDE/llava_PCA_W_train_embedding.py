import argparse
import torch
import os
from tqdm import tqdm
import pickle
import json
from PIL import Image
from llava_llama_2.utils import get_model
from llava_llama_2_utils import prompt_wrapper
import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="/data/huangyoucheng/mm-safety/ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")

    parser.add_argument("--attacked_image_fold", type=str, default='/data/huangyoucheng/mm-safety/visual_constrained_llava_train')
    parser.add_argument("--raw_image_fold", type=str, default='../val2017')
    parser.add_argument("--output_fold", type=str, default='llava_transfer_hidden_f_without_t')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_data_file", type=str,
                        default='../hh_harmless/train_filtered.jsonl', )
    parser.add_argument("--test_data_name", type=str,
                        default='hh_harmless', )
    args = parser.parse_args()
    return args


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')
args = parse_args()

print('model = ', args.model_path)

tokenizer, model, image_processor, model_name = get_model(args)
model.eval()
print('[Initialization Finished]\n')


def get_embeddings(file_prefix, file_name, file_suffix):
    img = load_image(os.path.join(file_prefix, file_name + file_suffix))
    img = image_processor.preprocess(img, return_tensors='pt')['pixel_values']
    img = model.encode_images(img.half())
    return img

def get_hidden(prompt, file_prefix, file_name, file_suffix):
    img = load_image(os.path.join(file_prefix, file_name + file_suffix))
    img = image_processor.preprocess(img, return_tensors='pt')['pixel_values']
    _ = model(input_ids=prompt, images=img.half(),  return_dict=True, output_hidden_states=True)
    return torch.stack([torch.stack(_.hidden_states[1:])[:, 0, -1, :]])


if  args.test_data_name == "do-not-answer":
    df = pd.read_csv(args.test_data_file)
    harmful_types = set(df["types_of_harm"])
    harmful_types.remove('Treat Chatbot as a Human')
    harmful_types = sorted(harmful_types)
    querys = df[df["types_of_harm"] != None]['question'].tolist()
elif args.test_data_name == "derogatory_corpus":
    df = pd.read_csv(args.test_data_file)
    querys = df.values.tolist()
    querys = [item[0] for item in querys]
elif args.test_data_name == "RealToxicityPrompt":
    querys = []
    with open(args.test_data_file, 'r', encoding='utf-8') as file:
        # 逐行读取文件

        for line in file:
            # 解码每行并将其转换为JSON对象
            json_obj = json.loads(line)
            # 将JSON对象添加到列表中
            querys.append(json_obj['prompt']['text'])
elif args.test_data_name == "hh_harmless":
    querys = []
    with open(args.test_data_file, 'r', encoding='utf-8') as file:
        # 逐行读取文件
        for line in file:
            # 解码每行并将其转换为JSON对象
            json_obj = json.loads(line)
            # 将JSON对象添加到列表中
            querys.append(json_obj['prompt'])
elif args.test_data_name == "safe_rlhf":
    querys = []
    with open(args.test_data_file, 'r', encoding='utf-8') as file:
        # 逐行读取文件
        for line in file:
            # 解码每行并将其转换为JSON对象
            json_obj = json.loads(line)
            # 将JSON对象添加到列表中
            querys.append(json_obj['prompt'])
elif args.test_data_name == "harmful_dataset":
    querys = []
    with open(args.test_data_file, 'r', encoding='utf-8') as file:
        # 逐行读取文件
        for line in file:
            # 解码每行并将其转换为JSON对象
            json_obj = json.loads(line)
            # 将JSON对象添加到列表中
            querys.append(json_obj['prompt'])
out = []
raw_image_files = os.listdir(args.raw_image_fold)
querys = sorted(querys)
raw_image_files = sorted(raw_image_files)
_t = torch.LongTensor([[29871]]).to(model.device)
os.makedirs(args.output_fold, exist_ok=True)
with torch.no_grad():
    for index in tqdm(range(5000)):
        user_message = querys[index]
        image_file = raw_image_files[index]

        text_prompt_template = prompt_wrapper.prepare_text_prompt(user_message)
        prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device).input_ids[0]
        # prompt = torch.cat((prompt, _t), dim=-1)
        image_file = image_file.split('.')[0]
        assert (image_file + '.jpg') in raw_image_files
        raw_img = get_hidden(prompt, args.raw_image_fold, image_file, '.jpg').half()
        pickle.dump(
            raw_img.cpu(),
            open(os.path.join(args.output_fold, image_file + '.pkl'), 'wb')
        )
