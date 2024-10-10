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
import pandas as pd
torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="[your path]/eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")

    parser.add_argument("--mode", type=str, default='VisualChatBot',
                        choices=[ "TextOnly", "VisualChatBot"],
                        help="Inference Mode: TextOnly: Text model only (Vicuna) \n VisualChatBot: Vision model + Text model (MiniGPT4) ")

    parser.add_argument("--raw_image_fold", type=str, default="../images/val2017")
    parser.add_argument("--output_fold", type=str, default="embeddings/minigpt_transfer_hidden")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_data_file", type=str,
                        default='../Queries_and_corpus/hh_train/train_filtered.jsonl', )
    parser.add_argument("--test_data_name", type=str,
                        default='hh_harmless', )
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
cfg = Config(args)
print(args)

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

prefix = prompt_wrapper.minigpt4_chatbot_prompt


model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print('Initialization Finished')

out = []
raw_image_files = os.listdir(args.raw_image_fold)
querys = sorted(querys)
raw_image_files = sorted(raw_image_files)
os.makedirs(args.output_fold, exist_ok=True)
with torch.no_grad():
    for index in tqdm(range(5000)):
        image_file = raw_image_files[index].split('.')[0]
        assert (image_file + '.jpg') in raw_image_files
        raw_img = Image.open(os.path.join(args.raw_image_fold, image_file + '.jpg')).convert('RGB')
        raw_img = [vis_processor(raw_img).unsqueeze(0).to(model.device)]

        text_prompt = prefix % ('Please extend the following sentence: %s')

        benign_prompt = prompt_wrapper.Prompt(model=model, img_prompts=[raw_img])
        # print([text_prompt % querys[index]])
        benign_prompt.update_text_prompt(text_prompts=[text_prompt % querys[index]], sign=False)

        raw_img_emb = torch.stack([torch.stack(model.llama_model.model(inputs_embeds=benign_prompt.context_embs[0], return_dict=True, output_hidden_states=True).hidden_states[1:])[:, 0, -1, :]]).cpu().half()
        img_embs = raw_img_emb
        pickle.dump(img_embs, open(os.path.join(args.output_fold, image_file + '.pkl'), 'wb'))
