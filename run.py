from my_function import *
import numpy as np
import lightgbm as lgb
import copy
import argparse
import random
import matplotlib
from pylab import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from tqdm import tqdm
import torch
import copy
import main
# import main_without_Daul
# import main_without_DWR
# import main_without_Daul_DWR
# import main_without_SSCC
# import main_withGCNR
# 更换绘图库
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-f', dest="fold", type=int)
results = parser.parse_args()
fold = results.fold  # 脚本的第一个参数指示当前运行的是第几折

n = 708  # n是药物数据
m = 1512  # m是靶点数量
# 读取数据
index_1 = np.loadtxt('divide_result/index_1.txt')  # 10折交叉验证中1的编号
index_0 = np.loadtxt('divide_result/index_0.txt')  # 10折交叉验证中0的编号
index = np.hstack((index_1, index_0))
# 索引矩阵
index = np.array(index)

# 药物0-708，靶标709-1512
drug_feature = np.loadtxt('training_result/embedding' + str(fold) + '.txt')[0:n, :]  # 前n行(0到n-1行)是学到的药物的特征向量
target_feature = np.loadtxt('training_result/embedding' + str(fold) + '.txt')[n:, :]  # 后

label = np.loadtxt('Data/DTI_708_1512.txt')

# 获得训练集与测试集的index
idx = copy.deepcopy(index)
test_index = copy.deepcopy(idx[fold])
test_index = np.array(test_index)
if test_index.ndim == 0:
    test_index = np.array([test_index])
idx = np.delete(idx, fold, axis=0)
train_index = idx.flatten()
insersection = np.intersect1d(test_index, train_index)

np.random.seed(10)
np.random.shuffle(train_index)

# 获得 （测试集 与 训练集） 的 （输入向量 与 标签）
test_input, test_output = get_feature(drug_feature, target_feature, label, test_index, n, m)
train_input, train_output = get_feature(drug_feature, target_feature, label, train_index, n, m)


# 构建SVM| 训练| 预测
print('start training')
lgb_train = lgb.Dataset(train_input, train_output)
lgb_eval = lgb.Dataset(test_input, test_output, reference=lgb_train)
# lightgbm的参数,binary用二元交叉熵损失函数
params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary',  # 目标二分类这里没问题
    'metric': {'average_precision'},  # 评估函数
    'is_unbalance': 'true',  # 针对数据集不平衡的情况进行的优化
    'num_leaves': 80,  # 叶子节点数
    'learning_rate': 0.02,  # 学习率
    'feature_fraction': 0.9,  # 建树的特征选择比例
    'bagging_fraction': 1,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 0,  # 显示模式
    'train_metric': 'true',
    "device": "gpu"
}
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                # early_stopping_rounds=2,
                )
y_pred = gbm.predict(test_input, num_iteration=1000)
t, f, p, r = tpr_fpr_precision_recall(test_output, y_pred)
print(auc(f, t))
print(auc(r, p) + r[0] * p[0])
print('The auc of prediction is:', roc_auc_score(test_output, y_pred))

np.savetxt('training_result/training' + str(fold) + '.txt', y_pred)
np.savetxt('training_result/training_test_index' + str(fold) + '.txt', test_index)


DTI = np.loadtxt("Data/DTI_708_1512.txt")  # 标签矩阵
index1 = np.loadtxt('divide_result/index_1.txt')
index0 = np.loadtxt('divide_result/index_0.txt')
index = np.hstack((index1, index0))
drug_num = DTI.shape[0]  # 708个药物
protein_num = DTI.shape[1]  # 1512个靶点
score = np.zeros(DTI.shape)

mask = np.zeros(DTI.shape)  # 用于标识是否是该折的数据

pre = np.loadtxt('training_result/training' + str(fold) + '.txt')
idx = index[fold, :]
for i in range(len(idx)):
    d = int(idx[i] / protein_num)
    p = int(idx[i] % protein_num)
    score[d, p] += pre[i]
    mask[d, p] = 1

DTI = DTI.tolist()
score = score.tolist()
# mask=mask.tolist()


auc_list = []
aupr_list = []
tpr_list = []
fpr_list = []
recall_list = []
precision_list = []
c = 0
for i in tqdm(range(drug_num)):  # 针对每一个药物而言
    if np.sum(np.array(DTI[i])[mask[i] == 1]) == 0:
        c += 1
        continue
    else:
        tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(DTI[i])[mask[i] == 1],
                                                                   np.array(score[i])[mask[i] == 1])
        fpr_list.append(fpr1)
        tpr_list.append(tpr1)
        precision_list.append(precision1)
        recall_list.append(recall1)
        auc_list.append(auc(fpr1, tpr1))
        aupr_list.append(auc(recall1, precision1) + recall1[0] * precision1[0])

tpr = equal_len_list(tpr_list)
fpr = equal_len_list(fpr_list)
precision = equal_len_list(precision_list)
recall = equal_len_list(recall_list)
tpr = np.array(tpr)
fpr = np.array(fpr)
precision = np.array(precision)
recall = np.array(recall)

tpr_mean = np.mean(tpr, axis=0)
fpr_mean = np.mean(fpr, axis=0)
recall_mean = np.mean(recall, axis=0)
precision_mean = np.mean(precision, axis=0)
AUC = auc(fpr_mean, tpr_mean)
AUPR = auc(recall_mean, precision_mean) + recall_mean[0] * precision_mean[
    0]
np.savetxt('10curves/tpr_mean_' + str(fold) + '.txt', tpr_mean)
np.savetxt('10curves/fpr_mean_' + str(fold) + '.txt', fpr_mean)
np.savetxt('10curves/recall_mean_' + str(fold) + '.txt', recall_mean)
np.savetxt('10curves/precision_mean_' + str(fold) + '.txt', precision_mean)

print('The auc of prediction is:%.4f' % AUC)
print('The aupr of prediction is:%.4f' % AUPR)
# 画ROC曲线

plt.figure()
plt.plot(fpr_mean, tpr_mean, label='ROC(AUC = %0.4f)' % AUC)
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend(loc="lower right")
plt.savefig('10fold_auc_aupr/' + str(fold) + '_ROC.jpg')
plt.show()

# 画PR曲线

plt.figure()
plt.plot(recall_mean, precision_mean, label='PR(AUPR = %0.4f)' % AUPR)
plt.title('PR curve')
plt.xlabel("RECALL")
plt.ylabel("PRECISION")
plt.legend(loc="lower right")
plt.savefig('10fold_auc_aupr/' + str(fold) + '_PR.jpg')
plt.show()

with open('10fold_auc_aupr/' + str(fold) + '_metrics.txt', 'w') as f:
    print('AUC:%.6f ' % AUC, 'AUPR: %.6f' % AUPR, file=f)


# DTI = np.loadtxt("Data/DTI_708_1512.txt")  # 标签矩阵
# index1 = np.loadtxt('divide_result/index_1.txt')
# index0 = np.loadtxt('divide_result/index_0.txt')
# index = np.hstack((index1, index0))
# drug_num = DTI.shape[0]  # 708个药物
# protein_num = DTI.shape[1]  # 1512个靶点
# score = np.zeros(DTI.shape)
# for f in tqdm(range(10)):
#     # pre = np.loadtxt('../../training_result/training' + str(f) + '.txt')
#     pre = np.loadtxt('training_result/training' + str(f) + '.txt')
#     idx = index[f, :]
#     # pre = line_normalize(pre)
#     # pre = pre[:, 1]
#     for i in range(len(idx)):
#         d = int(idx[i] / protein_num)
#         p = int(idx[i] % protein_num)
#         score[d, p] += pre[i]
#
# # 计算召回率
# TOPK_precent = 0.30
# TOPK = int(protein_num * TOPK_precent)  # 取TOPK个候选靶点
#
# all_drugs_recall = []
#
# score_tensor = torch.tensor(score)
# # 预测分数 预测靶点下标
# # candidate_target_score.shape=708,30
# # candidate_target_index.shape=708,30
# candidate_target_score, candidate_target_index = score_tensor.topk(k=TOPK, dim=1)
# candidate_target_score = np.array(candidate_target_score)
# candidate_target_index = np.array(candidate_target_index)
#
# c = 0
# for i in tqdm(range(drug_num)):  # 针对每一个药物而言
#     if np.sum(DTI[i]) == 0:  # 如果该药物不和任何靶点相互作用，则跳过
#         c += 1
#         continue
#     else:
#         drug_interaction_profile = copy.deepcopy(DTI[i])
#         TP_FN = np.sum(drug_interaction_profile)  # 召回率的分母
#         predict_target_index = copy.deepcopy(candidate_target_index[i])
#         ground_truth_target_index = np.where(drug_interaction_profile == 1)[0]
#         TP_targets = np.intersect1d(predict_target_index, ground_truth_target_index)
#         TP = TP_targets.shape[0]
#         temp_recall = TP / TP_FN
#         # all_drugs_recall=np.concatenate((all_drugs_recall,temp_recall))
#         all_drugs_recall.append(temp_recall)
#
#         # tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(DTI[i]), np.array(score[i]))
#
# all_drugs_recall = np.array(all_drugs_recall)
# average_all_drugs_recall = np.mean(all_drugs_recall)
# print("TOP percent:%f TOPK:%d average_recall:%.5f" % (TOPK_precent, TOPK, average_all_drugs_recall))
#
# tpr = equal_len_list(tpr_list)
# fpr = equal_len_list(fpr_list)
# precision = equal_len_list(precision_list)
# recall = equal_len_list(recall_list)
# tpr = np.array(tpr)
# fpr = np.array(fpr)
# precision = np.array(precision)
# recall = np.array(recall)
#
# tpr_mean = np.mean(tpr, axis=0)
# fpr_mean = np.mean(fpr, axis=0)
# recall_mean = np.mean(recall, axis=0)
# precision_mean = np.mean(precision, axis=0)
# AUC = auc(fpr_mean, tpr_mean)
# AUPR = auc(recall_mean, precision_mean) + recall_mean[0] * precision_mean[
#     0]  # 第(recall_mean[0],precision_mean[0])点的P值最高，R值最低，也就是PR曲线最左端的点
# print('The auc of prediction is:%.4f' % AUC)
# print('The aupr of prediction is:%.4f' % AUPR)

print('end')
