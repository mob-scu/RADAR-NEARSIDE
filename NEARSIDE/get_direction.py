
import os
import pickle
import torch
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dir', type=str, default="embeddings/llava_hh_train")
parser.add_argument('--save_path', type=str, default="directions/llava_hh_train.pkl")
args = parser.parse_args()

emb_files = os.listdir(args.embedding_dir)
emb_files = sorted(emb_files)

all_emb = []
for emb_file in tqdm(emb_files):
    with open(os.path.join(args.embedding_dir, emb_file), "rb") as f:
        all_emb.append(pickle.load(f))
# [attach_img, raw_img]
all_emb = torch.stack(all_emb).float()
# 得到方向d，这里的d是多个patch的 [p, dim]
direction = all_emb[:, 0, ...] - all_emb[:, 1, ...]
direction = direction.mean(dim=0)  # 在样本层面做mean
mag = torch.norm(direction, dim=-1, keepdim=True)  # 这里得到norm，也是多个patch的 [p]
direction = direction / mag
# # 在所有样本上做mean
# # 计算每一个攻击样本在d上的投影（除以norm），以每个patch为单位
margin_attack = (torch.einsum('bpd,pd->bp', all_emb[:, 0, ...], direction)).mean(dim=0)
# 计算每一个原始样本在d上的投影（除以norm），以每个patch为单位
margin_emb = (torch.einsum('bpd,pd->bp', all_emb[:, 1, ...], direction)).mean(dim=0)
# margin取中间值，以每个patch为单位
margin = (margin_attack + margin_emb) / 2

with open(args.save_path, 'wb') as f:
    direction = {
        "direction": direction,
        "mag": mag,
        "margin": margin
    }
    pickle.dump(direction, f)
