import argparse
import torch
import os
from tqdm import tqdm
import pickle
import json
from PIL import Image
from llava_llama_2.utils import get_model
from llava_llama_2_utils import prompt_wrapper


def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="/data/huangyoucheng/mm-safety/ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")

    parser.add_argument("--sign", type=bool, default=True, help="specify the gpu to load the model.")
    parser.add_argument("--list_path", type=str, default='../llava_attack_success_test_dc_demo_un')
    parser.add_argument("--raw_image_fold", type=str, default='../test2017')
    parser.add_argument("--output_fold", type=str, default='llava_dc_un_test_hidden_f')
    parser.add_argument("--batch_size", type=int, default=1)

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
as_qa = []
with open(os.path.join(args.list_path, 'success_qa.json'), 'r') as f:
    as_qa = json.load(f)



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

if args.list_path == '../llava_attack_success_train_hh':
    attacked_image_fold = '/data/huangyoucheng/mm-safety/visual_constrained_llava_train'
elif args.list_path == '../llava_attack_success_test_hh':
    attacked_image_fold = '/data/huangyoucheng/mm-safety/visual_constrained_llava_test'
elif args.list_path == '../llava_attack_success_test_dc_demo_32':
    attacked_image_fold = '/data/huangyoucheng/mm-safety/results_llava_llama_v2_demo_constrained_32'
elif args.list_path == '../llava_attack_success_test_dc_demo_16':
    attacked_image_fold = '/data/huangyoucheng/mm-safety/visual_llava_llama_v2_demo'
elif args.list_path == '../llava_attack_success_test_dc_demo_64':
    attacked_image_fold = '/data/huangyoucheng/mm-safety/results_llava_llama_v2_demo_constrained_64'
elif args.list_path == '../llava_attack_success_test_dc_demo_un':
    attacked_image_fold = '/data/huangyoucheng/mm-safety/results_llava_llama_v2_demo_unconstrained'
elif args.list_path == '../llava_attack_success_test_sr':
    attacked_image_fold = '/data/huangyoucheng/mm-safety/results_llava_llama_constrained_32_sr_train_filtered'
elif args.list_path == '../llava_attack_success_test_hd':
    attacked_image_fold = '/data/huangyoucheng/mm-safety/results_llava_llama_constrained_32_harmful_train_filtered'
print(args)
print(attacked_image_fold)
print(args.sign)
out = []
attacked_image_files = [item['img'] for item in as_qa]
raw_image_files = os.listdir(args.raw_image_fold)
attacked_image_files = sorted(attacked_image_files)
attacked_image_files = [i for n, i in enumerate(attacked_image_files) if i not in attacked_image_files[:n]]
os.makedirs(args.output_fold, exist_ok=True)
_t = torch.LongTensor([[29871]]).to(model.device)
with torch.no_grad():
    count = 0
    for item in tqdm(as_qa):
        user_message = item['query']
        text_prompt_template = prompt_wrapper.prepare_text_prompt(user_message)
        prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device).input_ids[0]
        if args.sign:
            prompt = torch.cat((prompt, _t), dim=-1)
        image_file = item['img']

        image_file = image_file.split('.')[0]
        assert (image_file + '.jpg') in raw_image_files
        attacked_img = get_hidden(prompt, attacked_image_fold, image_file, '.bmp').half()
        raw_img = get_hidden(prompt, args.raw_image_fold, image_file, '.jpg').half()
        pickle.dump(
            torch.cat([attacked_img, raw_img], dim=0).cpu(),
            open(os.path.join(args.output_fold, image_file + '.pkl'), 'wb')
        )
        count += 1
    print(count)