import torch
from numpy import *;
import numpy as np;
from math import *;
import torch.nn as nn
import copy


def null_list(num):
    lis = []
    for i in range(num): lis.append([])
    return lis


def laplacians(A):
    n = A.shape[0]  # m = n = 2220
    m = A.shape[1]
    A1 = A + np.eye(A.shape[0])  # 为邻接矩阵加自连
    D = np.sum(A1, axis=1)  # 计算每一行的和 D.shape=2220,1 其中的每一个元素代表结点的度
    A_L = np.zeros(A.shape)
    for i in range(n):
        for j in range(m):
            if i == j and D[i] != 0:
                A_L[i, j] = 1
            elif i != j and A1[i, j] != 0:
                A_L[i, j] = (-1) / sqrt(D[i] * D[j])
            else:
                A_L[i, j] = 0
    return A_L


def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)


# 此时adj_matrix应输入张量
def normalized_laplacian_gpu(adj_matrix):
    R = torch.sum(adj_matrix, dim=1)
    R_sqrt = 1 / torch.sqrt(R)
    D_sqrt = torch.diag_embed(R_sqrt)
    I = torch.eye(adj_matrix.shape[0])
    return I - torch.mm(torch.mm(D_sqrt, adj_matrix), D_sqrt)


def max_min_normalize(a):  # 矩阵归一化
    sum_of_line = np.sum(a, axis=1)
    line = a.shape[0]
    row = a.shape[1]
    i = 0
    while i < line:
        j = 0
        while j < row:
            if sum_of_line[i] != 0:
                max = np.max(a[i])
                min = np.min(a[i])
                a[i, j] = (a[i, j] - min) / (max - min)
            j = j + 1
        i = i + 1
    return a


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


def equal_len_list(a):
    row_len = []
    for i in a:
        row_len.append(len(i))
    min_len = min(row_len)
    equal_len_a = []
    for i in a:
        tem_list = []
        multi = len(i) / min_len
        for j in range(min_len):
            tem_list.append(i[int(j * multi)])
        equal_len_a.append(tem_list)
    return equal_len_a


def remove_ele(a, x):
    b = []
    for i in a:
        flag = 1
        while flag == 1:
            if x in i:
                i.remove(x)
            else:
                b.append(i)
                flag = 0
    return b

def get_feature(drug_feature, target_feature, label, index, drug_num, target_num):
    # print("Index shape:", index.shape)
    # print("Index values:", index)
    input_data = []
    output_data = []
    if index.ndim == 0:  # Check if index is 0-dimensional
        index = np.array([index])  # Convert index to a 1-dimensional array
    for i in range(np.size(index)):
        drug = int(index[i] / target_num)
        target = int(index[i] % target_num)
        feature = np.hstack((drug_feature[drug], target_feature[target]))
        input_data.append(feature.tolist())
        output_data.append(label[drug, target])
    return np.array(input_data), np.array(output_data)


def cosine_similarity(matrix1, matrix2):
    """
    计算两个矩阵的余弦相似度

    参数:
    matrix1: 第一个矩阵，numpy数组形式
    matrix2: 第二个矩阵，numpy数组形式

    返回值:
    余弦相似度
    """

    # 将PyTorch张量转换为NumPy数组
    if isinstance(matrix1, torch.Tensor):
        matrix1 = matrix1.detach().cpu().numpy()
    if isinstance(matrix2, torch.Tensor):
        matrix2 = matrix2.detach().cpu().numpy()

    dot_product = np.dot(matrix1.flatten(), matrix2.flatten())
    norm_matrix1 = np.linalg.norm(matrix1)
    norm_matrix2 = np.linalg.norm(matrix2)

    similarity = dot_product / (norm_matrix1 * norm_matrix2)
    return similarity


def pearson_correlation(matrix1, matrix2):
    """
    计算两个矩阵之间的皮尔逊相关系数。

    参数:
    matrix1, matrix2: numpy数组，两个形状相同的矩阵。

    返回:
    correlation: float，两个矩阵之间的皮尔逊相关系数。
    """
    # 确保输入矩阵的形状相同
    assert matrix1.shape == matrix2.shape, "输入矩阵形状不一致"
    # 将PyTorch张量转换为NumPy数组
    if isinstance(matrix1, torch.Tensor):
        matrix1 = matrix1.detach().cpu().numpy()
    if isinstance(matrix2, torch.Tensor):
        matrix2 = matrix2.detach().cpu().numpy()

    # 计算皮尔逊相关系数
    flattened1 = matrix1.flatten()
    flattened2 = matrix2.flatten()
    correlation = np.corrcoef(flattened1, flattened2)[0, 1]

    return correlation


def get_feature_denoise(drug_feature, target_feature, label, index, drug_num, target_num, DTI_WKNKN):
    # 根据WKNKN后的DTI矩阵查找，如果WKNKN后的DTI矩阵中的药物靶点对是小数，我们就不加入训练
    input = []
    output = []
    for i in range(index.shape[0]):
        drug = int(index[i] / target_num)
        target = int(index[i] % target_num)
        if (DTI_WKNKN[drug][target] != 0) and (DTI_WKNKN[drug][target] != 1):  # 如果在WKNKN矩阵中的对应值不是0或1，那么我们就不加入训练
            continue
        feature = np.hstack((drug_feature[drug], target_feature[target]))
        input.append(feature.tolist())
        output.append(label[drug, target])
    return np.array(input), np.array(output)


def line_normalize(A):  # 行归一化
    sum_of_line = np.sum(A, axis=1)
    line = A.shape[0]
    row = A.shape[1]
    i = 0
    while i < line:
        j = 0
        while j < row:
            if sum_of_line[i] != 0:
                A[i, j] = A[i, j] / sum_of_line[i]
            j = j + 1
        i = i + 1
    return A


# true是标签，pred是预测值
def tpr_fpr_precision_recall(true, pred):
    num = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    index = list(reversed(np.argsort(pred)))
    tpr = []
    fpr = []
    precision = []
    recall = []
    for i in range(pred.shape[0]):
        if true[int(index[i])] == 1:
            tp += 1
        else:
            fp += 1
        if np.sum(true) == 0:
            tpr.append(0)
            fpr.append(0)
            precision.append(0)
            recall.append(0)
        else:
            tpr.append(tp / np.sum(true))
            fpr.append(fp / (true.shape[0] - np.sum(true)))
            precision.append(tp / (tp + fp))
            recall.append(tp / np.sum(true))
    return tpr, fpr, precision, recall


def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

# 对称归一化的laplacian矩阵,输入是邻接矩阵，输出是对称归一化后的拉普拉斯矩阵
def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt = 1 / np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

# 线性缩放稀疏化
def sparse_matrix(similarity_matrix, p):
    length = similarity_matrix.shape[0]
    N = np.zeros((length, length))

    max_sim = np.max(similarity_matrix)
    min_sim = np.min(similarity_matrix)

    for i in range(length):
        pNeighborsofi = pNeighbors(i, similarity_matrix, p)
        for j in range(length):
            pNeighborsofj = pNeighbors(j, similarity_matrix, p)

            similarity_value = similarity_matrix[i][j]
            normalized_value = (similarity_value - min_sim) / (max_sim - min_sim)  # 归一化

            if (j not in pNeighborsofi) and (i not in pNeighborsofj):
                N[i][j] = 0  # 无邻居
            else:
                N[i][j] = normalized_value  # 弱邻居，同样使用归一化后的相似度值

    similarity_matrix_after_sparse = np.multiply(similarity_matrix, N)
    similarity_matrix_after_sparse += np.eye(length)
    return similarity_matrix_after_sparse


def pNeighbors(node, matrix, K):  # 根据相似性矩阵返回K近邻
    KknownNeighbors = np.array([])
    featureSimilarity = copy.deepcopy(matrix[node])  # 在相似性矩阵中取出第node行
    featureSimilarity[node] = -100  # 排除自身结点,使相似度为-100
    KknownNeighbors = featureSimilarity.argsort()[::-1]  # 按照相似度降序排序
    KknownNeighbors = KknownNeighbors[:K]  # 返回前K个结点的下标
    return KknownNeighbors
