import numpy as np
from tqdm import tqdm
import my_function


def Jaccard_similarity(A):
    # B是返回的杰卡德相似性矩阵
    B = np.zeros((A.shape[0], A.shape[0]))
    for i in tqdm(range(B.shape[0])):
        for j in range(i + 1, B.shape[1]):  # 只做上三角部分
            # 下面的代码计算药物i和药物j的相似性
            if np.sum(A[i]) == 0 and np.sum(A[j]) == 0:  # 如果两个药物不和任何靶点相互作用，则两个药物的相似度为0
                B[i][j] = 0
            else:
                jiaoji = 0
                bingji = 0
                # 计算A[i]和A[j]的交集和并集
                for k in range(A.shape[1]):
                    if A[i][k] == 1 and A[j][k] == 1:
                        jiaoji += 1
                        bingji += 1
                    elif A[i][k] == 1 or A[j][k] == 1:
                        bingji += 1
                B[i][j] = jiaoji / bingji
    # 此时B只是上三角矩阵，将上三角矩阵转为对称阵
    # 将主对角元素置为1
    # 因为有些药物不和任何靶点相互作用，但是自己和自己的相似度肯定是1
    row, col = np.diag_indices_from(B)
    B[row, col] = 1
    B += B.T - np.diag(B.diagonal())

    # print(B.T==B)

    return B


def max_min_normalize(a):  # 矩阵归一化
    sum_of_line = np.sum(a, axis=1)
    line = a.shape[0]
    row = a.shape[1]
    i = 0
    a = a.tolist()
    a_n = []
    for i in range(len(a)):
        if sum_of_line[i] != 0:
            max = np.max(np.array(a[i]))
            min = np.min(np.array(a[i]))
            t = []
            for j in range(len(a[0])):
                t.append((a[i][j] - min) / (max - min))
        else:
            t = []
            for j in range(len(a[0])):
                t.append(0)
        a_n.append(t)
    return np.array(a_n)


def min_max_normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    return (matrix - min_val) / (max_val - min_val)


def diffusion_rwr(A, maxiter, restart_prob):
    n = A.shape[0]  # 获取邻接矩阵的行数

    # 为孤立节点添加自环
    A = A + np.diag(np.sum(A, axis=1) == 0)

    # 规范化邻接矩阵
    row_sums = np.sum(A, axis=1)
    P = A / row_sums[:, np.newaxis]  # 转移概率矩阵

    # 初始化变量
    restart = np.eye(n)
    Q = np.eye(n)

    for _ in range(maxiter):
        Q_new = (1 - restart_prob) * P @ Q + restart_prob * restart
        delta = np.linalg.norm(Q - Q_new, 'fro')
        Q = Q_new
        if delta < 1e-6:
            break

    return Q


if __name__ == "__main__":
    # 药物相似性矩阵融合
    drug_drug_interaction = np.loadtxt('Data/DDI_708_708.txt')  # 药物-药物相互作用矩阵
    drug_disease_association = np.loadtxt('Data/drug_disease_708_5603.txt')
    drug_sideeffect_association = np.loadtxt('Data/drug_sideeffect_708_4192.txt')
    drug_drug_chemistry_similarity = np.loadtxt('Data/SD_708_708.txt')
    # 前三个矩阵需要求jaccard相似度
    drug_drug_interaction_similarity = Jaccard_similarity(drug_drug_interaction)
    drug_disease_association_similarity = Jaccard_similarity(drug_disease_association)
    drug_sideeffect_association_similarity = Jaccard_similarity(drug_sideeffect_association)
    # 保存一下把，可能计算的比较慢
    np.save('Data/multi_similarity/drug_drug_interaction_similarity.npy', drug_drug_interaction_similarity)
    np.save('Data/multi_similarity/drug_disease_association_similarity.npy', drug_disease_association_similarity)
    np.save('Data/multi_similarity/drug_sideeffect_association_similarity.npy',
            drug_sideeffect_association_similarity)
    np.save('Data/multi_similarity/drug_drug_chemistry_similarity.npy', drug_drug_chemistry_similarity)
    # 方案1：取最大并归一化
    x1 = np.maximum(drug_drug_chemistry_similarity, drug_drug_interaction_similarity)
    x2 = np.maximum(x1, drug_disease_association_similarity)
    drug_fusion_similarity = np.maximum(x2, drug_sideeffect_association_similarity)
    drug_fusion_similarity = max_min_normalize(drug_fusion_similarity)
    np.savetxt('Data/multi_similarity/drug_fusion_similarity_708_708.txt', drug_fusion_similarity)

    # 靶点相似性矩阵融合
    target_disease_association = np.loadtxt('Data/target_disease_1512_5603.txt')
    target_target_interaction = np.loadtxt('Data/TTI_1512_1512.txt')
    target_target_sequence_similarity = np.loadtxt('Data/ST_1512_1512.txt')
    # 前两个矩阵需要求jaccard相似度
    target_disease_association_similarity = Jaccard_similarity(target_disease_association)
    target_target_interaction_similarity = Jaccard_similarity(target_target_interaction)
    np.save('Data/multi_similarity/target_disease_association_similarity.npy',
            target_disease_association_similarity)
    np.save('Data/multi_similarity/target_target_interaction_similarity.npy',
            target_target_interaction_similarity)
    np.save('Data/multi_similarity/target_target_sequence_similarity.npy', target_target_sequence_similarity)

    # 方案1：取最大并归一化
    y1 = np.maximum(target_target_sequence_similarity, target_disease_association_similarity)
    target_fusion_similarity = np.maximum(y1, target_target_interaction_similarity)
    target_fusion_similarity = max_min_normalize(target_fusion_similarity)
    np.savetxt('Data/multi_similarity/target_fusion_similarity_1512_1512.txt', target_fusion_similarity)

    print('end')
