
import os
import pickle
from tqdm import tqdm
import torch
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--direction_file", type=str, default="minigpt_hh_train_direction_f.pkl")
parser.add_argument("--test_fold", type=str, default="minigpt_hd_test_hidden_f")
args = parser.parse_args()

direction = pickle.load(open(args.direction_file, "rb"))
direction, mag, margin = direction['direction'], direction['mag'], direction['margin']

test_emb = []
emb_fold = args.test_fold
emb_files = os.listdir(emb_fold)
emb_files = sorted(emb_files)
for emb_file in tqdm(emb_files):
    with open(os.path.join(emb_fold, emb_file), "rb") as f:
        test_emb.append(pickle.load(f))

acc0, acc1 = [0 for _ in range(40)], [0 for _ in range(40)]
proj_att = []
proj_ben = []
test_emb = test_emb[:100]
for t_emb in test_emb:
    # [attack, raw, random_noise]
    # 计算该样本在direction上的投影（除以norm），以patch为单位
    projection = torch.einsum('bpd,pd->bp', t_emb.float(), direction)  # 3,p
    # proj_att.append(projection[0][39])
    # proj_ben.append(projection[1][39])
    judge_0 = projection[0] - margin
    judge_1 = projection[1] - margin
    # 综合判断（投票）
    for l in range(40):

        if judge_0[l].mean() > 0:  # attack 图片二分类，比margin要大为对，以每个patch为单位
            acc0[l] += 1
            # print(index)
            # print('****************')
            # print(projection[0])
            # print('============')
            # print(projection[1])
            # print('============')
            # print(margin)
        if judge_1[l].mean() < 0:  # raw 图片二分类，比margin要小为对，以每个patch为单位
            acc1[l] += 1


for l in range(40):
    print(f"layer{l}:")
    print(f"acc_attack:{acc0[l]}   {acc0[l] / len(test_emb) * 100:.2f}   acc_raw:{acc1[l]}  {acc1[l] / len(test_emb) * 100:.2f}")
    print(f"acc: {(acc0[l]+acc1[l]) / (2 * len(test_emb)) * 100:.2f}%")
print('\n')
# print(np.array(proj_att))
# print(np.array(proj_ben))
# print(np.array(margin[39]))
# result = []
# result.append(np.array(proj_att))
# result.append(np.array(proj_ben))
# result.append(np.array(margin[39]))
# os.makedirs(f'llava_hd_analysis', exist_ok=True)
#
# pickle.dump(result, open(os.path.join(f'llava_hd_analysis', f'projatt_projben_margin.pkl'), 'wb'))

exit()
#
# for t_emb in test_emb:
#     # [attack, raw, random_noise]
#     # 计算该样本在direction上的投影（除以norm），以patch为单位
#     projection = torch.einsum('bpd,pd->bp', t_emb.float(), direction)   # 3,p
#     judge_0 = projection[0] - margin
#     judge_1 = projection[1] - margin
#     # 综合判断（投票）
#     if judge_0.mean() > 0:  # attack 图片二分类，比margin要大为对，以每个patch为单位
#         acc0 += 1
#         # print(index)
#         # print('****************')
#         # print(projection[0])
#         # print('============')
#         # print(projection[1])
#         # print('============')
#         # print(margin)
#     if judge_1.mean() < 0:  # raw 图片二分类，比margin要小为对，以每个patch为单位
#         acc1 += 1
#     index += 1
# print(f"acc_attack:{acc0}   {acc0 / len(test_emb) * 100:.2f}   acc_raw:{acc1}  {acc1 / len(test_emb) * 100:.2f}")
# print(f"acc: {(acc0+acc1) / (2 * len(test_emb)) * 100:.2f}%")
