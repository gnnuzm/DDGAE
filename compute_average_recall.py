from evaluation_10curves import tpr_list, fpr_list, precision_list, recall_list
from my_function import *
import numpy as np
from pylab import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from tqdm import tqdm
import torch
import copy
from matplotlib import pyplot as plt

DTI = np.loadtxt("Data/DTI_708_1512.txt")  # 标签矩阵
index1 = np.loadtxt('divide_result/index_1.txt')
index0 = np.loadtxt('divide_result/index_0.txt')
index = np.hstack((index1, index0))
drug_num = DTI.shape[0]  # 708个药物
protein_num = DTI.shape[1]  # 1512个靶点
score = np.zeros(DTI.shape)
for f in tqdm(range(10)):
    pre = np.loadtxt('training_result/training' + str(f) + '.txt')
    idx = index[f, :]

    for i in range(len(idx)):
        d = int(idx[i] / protein_num)
        p = int(idx[i] % protein_num)
        score[d, p] += pre[i]


TOPK_precent = 0.05
TOPK = int(protein_num * TOPK_precent)  # 取TOPK个候选靶点

all_drugs_recall = []

score_tensor = torch.tensor(score)
# 预测分数 预测靶点下标
candidate_target_score, candidate_target_index = score_tensor.topk(k=TOPK, dim=1)
candidate_target_score = np.array(candidate_target_score)
candidate_target_index = np.array(candidate_target_index)

c = 0
for i in tqdm(range(drug_num)):  # 针对每一个药物而言
    if np.sum(DTI[i]) == 0:  # 如果该药物不和任何靶点相互作用，则跳过
        c += 1
        continue
    else:
        drug_interaction_profile = copy.deepcopy(DTI[i])
        TP_FN = np.sum(drug_interaction_profile)  # 召回率的分母
        predict_target_index = copy.deepcopy(candidate_target_index[i])
        ground_truth_target_index = np.where(drug_interaction_profile == 1)[0]
        TP_targets = np.intersect1d(predict_target_index, ground_truth_target_index)
        TP = TP_targets.shape[0]
        temp_recall = TP / TP_FN

        all_drugs_recall.append(temp_recall)


all_drugs_recall = np.array(all_drugs_recall)
average_all_drugs_recall = np.mean(all_drugs_recall)
print("TOP percent:%f TOPK:%d average_recall:%.5f" % (TOPK_precent, TOPK, average_all_drugs_recall))

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
    0]  # 第(recall_mean[0],precision_mean[0])点的P值最高，R值最低，也就是PR曲线最左端的点
print('The auc of prediction is:%.4f' % AUC)
print('The aupr of prediction is:%.4f' % AUPR)


