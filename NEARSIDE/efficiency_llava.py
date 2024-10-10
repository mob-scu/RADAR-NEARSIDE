import argparse
import torch
import os
from tqdm import tqdm
import pickle
import json
from PIL import Image
from llava_llama_2.utils import get_model
from llava_llama_2_utils import prompt_wrapper
import pickle
import torch
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser()

parser.add_argument("--model-path", type=str, default="/data/huangyoucheng/mm-safety/ckpts/llava_llama_2_13b_chat_freeze")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
parser.add_argument("--direction_file", type=str, default="llava_hh_train_direction_f.pkl")
parser.add_argument("--list_path", type=str, default="../llava_attack_success_test_hh")
# parser.add_argument("--test_fold", type=str, default="minigpt_hd_test_hidden_f")
args = parser.parse_args()
tokenizer, model, image_processor, model_name = get_model(args)

model.eval()
start_time = time.time()
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image
def get_hidden(prompt, file_prefix, file_name, file_suffix):
    img = load_image(os.path.join(file_prefix, file_name + file_suffix))
    img.save(os.path.join("eff_imgs", file_name + file_suffix))
    img = image_processor.preprocess(img, return_tensors='pt')['pixel_values']
    _ = model(input_ids=prompt, images=img.half(),  return_dict=True, output_hidden_states=True)
    return torch.stack([torch.stack(_.hidden_states[1:])[:, 0, -1, :]])

direction = pickle.load(open(args.direction_file, "rb"))
direction, mag, margin = direction['direction'], direction['mag'], direction['margin']

attacked_image_fold = '/data/huangyoucheng/mm-safety/visual_constrained_llava_test'
raw_image_fold = '/data/huangyoucheng/mm-safety/test2017'
as_qa = []
with open(os.path.join(args.list_path, 'success_qa.json'), 'r') as f:
    as_qa = json.load(f)
as_qa = as_qa[:10]
test_emb = []
_t = torch.LongTensor([[29871]]).to(model.device)
with torch.no_grad():
    for item in tqdm(as_qa):
        user_message = item['query']
        text_prompt_template = prompt_wrapper.prepare_text_prompt(user_message)
        prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device).input_ids[0]
        prompt = torch.cat((prompt, _t), dim=-1)
        image_file = item['img']
        if not os.path.exists(os.path.join(attacked_image_fold, image_file)):
            continue
        image_file = image_file.split('.')[0]
        attacked_img = get_hidden(prompt, attacked_image_fold, image_file, '.bmp').half()
        raw_img = get_hidden(prompt, raw_image_fold, image_file, '.jpg').half()
        test_emb.append(torch.cat([attacked_img, raw_img], dim=0).cpu())


acc0, acc1 = [0 for _ in range(40)], [0 for _ in range(40)]

for t_emb in test_emb:
    projection = torch.einsum('bpd,pd->bp', t_emb.float(), direction)  # 3,p
    judge_0 = projection[0] - margin
    judge_1 = projection[1] - margin
    l = 39
    if judge_0[l].mean() > 0:  # attack 图片二分类，比margin要大为对，以每个patch为单位
        acc0[l] += 1
    if judge_1[l].mean() < 0:  # raw 图片二分类，比margin要小为对，以每个patch为单位
        acc1[l] += 1
print(f"layer{l}:")
print(f"acc_attack:{acc0[l]}   {acc0[l] / len(test_emb) * 100:.2f}   acc_raw:{acc1[l]}  {acc1[l] / len(test_emb) * 100:.2f}")
print(f"acc: {(acc0[l]+acc1[l]) / (2 * len(test_emb)) * 100:.2f}%")
end_time = time.time()
print(f"时间: {end_time - start_time:.6f} 秒")
exit()




