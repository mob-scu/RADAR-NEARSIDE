import argparse
import os
import random
from tqdm import tqdm
import pickle
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

from minigpt_utils import prompt_wrapper
torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=5, help="specify the gpu to load the model.")
    parser.add_argument("--mode", type=str, default='VisualChatBot',
                        choices=[ "TextOnly", "VisualChatBot"],
                        help="Inference Mode: TextOnly: Text model only (Vicuna) \n VisualChatBot: Vision model + Text model (MiniGPT4) ")
    parser.add_argument("--sign", type=bool, default=True, help="specify the gpu to load the model.")
    parser.add_argument("--list_path", type=str, default='../minigpt_attack_success_test_hd')
    parser.add_argument("--raw_image_fold", type=str, default="../test2017")
    parser.add_argument("--output_fold", type=str, default="minigpt_hd_test_hidden_f")
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair " 
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

args = parse_args()

if args.list_path == '../minigpt_attack_success_train_hh':
    attacked_image_fold = '../visual_constrained_eps_32_hh_rlhf'
elif args.list_path == '../minigpt_attack_success_test_hh':
    attacked_image_fold = '../visual_constrained_minpgpt_test'
elif args.list_path == '../minigpt_attack_success_test_dc_demo_32':
    attacked_image_fold = '../results_minigpt_constrained_32_demo_fixed'
elif args.list_path == '../minigpt_attack_success_test_sr':
    attacked_image_fold = '../results_minigpt_constrained_32_sr_train_filtered_fixed'
elif args.list_path == '../minigpt_attack_success_test_hd':
    attacked_image_fold = '../results_minigpt_constrained_32_harmful_train_filtered_fixed'
print(attacked_image_fold)
print(args.sign)
cfg = Config(args)
print(args)
as_qa = []
with open(os.path.join(args.list_path, 'success_qa.json'), 'r') as f:
    as_qa = json.load(f)

prefix = prompt_wrapper.minigpt4_chatbot_prompt


model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print('Initialization Finished')

out = []
attacked_image_files = [item['img'] for item in as_qa]
raw_image_files = os.listdir(args.raw_image_fold)
attacked_image_files = sorted(attacked_image_files)
attacked_image_files = [i for n, i in enumerate(attacked_image_files) if i not in attacked_image_files[:n]]
os.makedirs(args.output_fold, exist_ok=True)
with torch.no_grad():
    for item in tqdm(as_qa[:500]):
        image_file = item['img'].split('.')[0]
        assert (image_file + '.jpg') in raw_image_files
        attacked_img = Image.open(os.path.join(attacked_image_fold, image_file + '.bmp')).convert('RGB')
        raw_img = Image.open(os.path.join(args.raw_image_fold, image_file + '.jpg')).convert('RGB')

        attacked_img = [vis_processor(attacked_img).unsqueeze(0).to(model.device)]
        raw_img = [vis_processor(raw_img).unsqueeze(0).to(model.device)]

        text_prompt = prefix % ('Please extend the following sentence: %s')

        attack_prompt = prompt_wrapper.Prompt(model=model, img_prompts=[attacked_img])
        benign_prompt = prompt_wrapper.Prompt(model=model, img_prompts=[raw_img])
        attack_prompt.update_text_prompt([text_prompt % item['query']], sign=args.sign)
        benign_prompt.update_text_prompt([text_prompt % item['query']], sign=args.sign)

        attacked_img_emb = torch.stack([torch.stack(model.llama_model.model(inputs_embeds=attack_prompt.context_embs[0], return_dict=True, output_hidden_states=True).hidden_states[1:])[:, 0, -1, :]]).cpu().half()
        raw_img_emb = torch.stack([torch.stack(model.llama_model.model(inputs_embeds=benign_prompt.context_embs[0], return_dict=True, output_hidden_states=True).hidden_states[1:])[:, 0, -1, :]]).cpu().half()
        img_embs = torch.cat([attacked_img_emb, raw_img_emb], dim=0)
        pickle.dump(img_embs, open(os.path.join(args.output_fold, image_file + '.pkl'), 'wb'))
