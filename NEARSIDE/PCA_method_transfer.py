import numpy as np
import os
import pickle
from tqdm import tqdm
import torch
import argparse
import torch
from sklearn.decomposition import PCA
from scipy.optimize import minimize

parser = argparse.ArgumentParser()
parser.add_argument("--direction_file", type=str, default='directions/llava_hh_train.pkl')

parser.add_argument("--test_fold_source", type=str, default="embeddings/llava_transfer_hidden")
parser.add_argument("--test_fold_target", type=str, default="embeddings/minigpt_transfer_hidden")

parser.add_argument("--train_fold", type=str, default="embeddings/llava_hh_train_hidden_f")
parser.add_argument("--test_fold", type=str, default="embeddings/minigpt_hh_test_hidden_f")

parser.add_argument("--dim", type=int, default=2048)

args = parser.parse_args()
print(args)
direction = pickle.load(open(args.direction_file, "rb"))
direction, mag, margin = direction['direction'], direction['mag'], direction['margin']
direction = direction * mag

t_emb = []
emb_fold = args.test_fold_target
emb_files = os.listdir(emb_fold)
emb_files = sorted(emb_files)
for emb_file in emb_files:
    with open(os.path.join(emb_fold, emb_file), "rb") as f:
        t_emb.append(pickle.load(f))
target_emb = t_emb
target_emb = torch.stack(target_emb)[:, :, -1,:]

t_emb = []
emb_fold = args.test_fold_source
emb_files = os.listdir(emb_fold)
emb_files = sorted(emb_files)
for emb_file in emb_files:
    with open(os.path.join(emb_fold, emb_file), "rb") as f:
        t_emb.append(pickle.load(f))
source_emb = t_emb
source_emb = torch.stack(source_emb)[:, :, -1,:]

X = np.array(source_emb.view(source_emb.shape[0], -1))

Y = np.array(target_emb.view(target_emb.shape[0], -1))


pca_s = PCA(n_components=args.dim)
pca_s.fit(X)
pca_t = PCA(n_components=args.dim)
pca_t.fit(Y)

# train pca model

X = pca_s.transform(X)
Y = pca_t.transform(Y)
direction = torch.tensor(pca_s.transform([direction[-1].view(-1)])[0])

# 最小二乘法
beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

# 约束法
# rho = 0.1  # Regularization parameter
# def objective(W, X, Y, rho):
#     W = W.reshape(X.shape[1], Y.shape[1])
#     term1 = np.sum(np.sqrt(np.sum((X @ W - Y)**2, axis=0)))
#     term2 = rho * np.sum(W**2)
#     return term1 + term2
# # 初始猜测
# W0 = np.random.randn(X.shape[1] * Y.shape[1])
#
# # 执行优化
# result = minimize(objective, W0, args=(X, Y, rho), method='L-BFGS-B')
#
# # 结果
# beta = result.x.reshape(X.shape[1], Y.shape[1])
# beta

all_emb = []
emb_fold = args.train_fold
emb_files = os.listdir(emb_fold)
emb_files = sorted(emb_files)
for emb_file in emb_files:
    with open(os.path.join(emb_fold, emb_file), "rb") as f:
        all_emb.append(pickle.load(f))
# 降维
# all_emb = [item[:, -1, :].view(item.shape[0], -1) for item in tqdm(all_emb)]
all_emb = [pca_s.transform(item[:, -1, :].view(item.shape[0], -1)) for item in all_emb]
all_emb = torch.tensor(np.array(all_emb))

# 映射
all_emb = torch.stack([torch.stack([torch.tensor(np.dot(item[0], beta)), torch.tensor(np.dot(item[1], beta))]) for item in all_emb])


direction_t = torch.tensor(np.dot(direction, beta))
mag = torch.norm(direction_t, dim=-1, keepdim=True)  # 这里得到norm，也是多个patch的 [p]
direction_t = direction_t / mag
margin_attack = torch.einsum('bp,p->b', all_emb[:, 0, ...], direction_t)
# 计算每一个原始样本在d上的投影（除以norm），以每个patch为单位
margin_emb = torch.einsum('bp,p->b', all_emb[:, 1, ...], direction_t)
# margin取中间值，以每个patch为单位
margin = (margin_attack + margin_emb) / 2
# margin = np.dot(margin, beta)




test_emb = []

emb_fold = args.test_fold
emb_files = os.listdir(emb_fold)
emb_files = sorted(emb_files)
for emb_file in emb_files:
    with open(os.path.join(emb_fold, emb_file), "rb") as f:
        test_emb.append(pickle.load(f))

acc0, acc1, acc2 = 0, 0, 0
for t_emb in tqdm(test_emb):
    t_emb = t_emb[:, -1]
    t_emb = torch.tensor(pca_t.transform(t_emb.view(t_emb.shape[0], -1)))
    projection = torch.einsum('bp,p->b', t_emb, direction_t) # 3,p
    judge_0 = projection[0] - margin
    judge_1 = projection[1] - margin
    # 综合判断（投票）
    if judge_0.mean() > 0:  # attack 图片二分类，比margin要大为对，以每个patch为单位
        acc0 += 1
    if judge_1.mean() < 0:  # raw 图片二分类，比margin要小为对，以每个patch为单位
        acc1 += 1
print(f"acc_attack:{acc0}   {acc0 / len(test_emb) * 100:.2f}   acc_raw:{acc1}  {acc1 / len(test_emb) * 100:.2f}")
print(f"acc: {(acc0+acc1) / (2 * len(test_emb)) * 100:.2f}%")


