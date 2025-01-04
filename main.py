import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp
from torch.nn import Linear
from torch_geometric.data import Data
import numpy as np
from my_function import *
from torch_geometric.nn import GCNConv
import copy
import sys
import argparse
import os

torch.autograd.set_detect_anomaly(True)

# @@@@@@@@@@@@@@@@@@@@@
parser = argparse.ArgumentParser()
parser.add_argument('-f', dest="fold", type=int)
results = parser.parse_args()
fold = results.fold  # 脚本的第一个参数指示当前运行的是第几折
# @@@@@@@@@@@@@@@@@@@@@@@@@@

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 获取GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCH = 4000  # 训练epoch数
BATCH_SIZE = 64  # 48*64
LR = 0.0001
n = 708  # 708个药物
m = 1512  # 1512个靶点

in_features = 2220  # m+n
out_features = 200  # ==> (m+n)xk  2220x200
# 图卷积第一层结束时 a=500
N_HID = 500

adjust_p_neighbors_parameters = True

lambda_l = 0.00001
lambda_d = 0.001
lambda_t = 0.001

alpha = 0.5
beta = 0.1
gamma = 0.01


# 2220 x2220
A = np.loadtxt('divide_result/A' + str(fold) + '.txt')  # 其中A有WKNKN的信息
X = np.loadtxt('divide_result/X' + str(fold) + '.txt')  # 其中X有融合相似度的信息

# GCNConv 默认有对称归一化邻接矩阵效果
edge_index_temp = sp.coo_matrix(A)
edge_weight = copy.deepcopy(edge_index_temp.data)  # 边权
edge_weight = torch.FloatTensor(edge_weight).to(device)  # 将numpy转为tensor 我们要利用的边权
edge_index_A = np.vstack((edge_index_temp.row, edge_index_temp.col))  # 提取的边[2,num_edges]
edge_index_A = torch.LongTensor(edge_index_A).to(device)  # 将numpy转为tensor 我们要利用的边的index

# SSCC稀疏化
if adjust_p_neighbors_parameters == True:
    Sd = np.loadtxt('Data/multi_similarity/drug_fusion_similarity_708_708.txt')
    St = np.loadtxt('Data/multi_similarity/target_fusion_similarity_1512_1512.txt')
    Sd_after_sparse = sparse_matrix(similarity_matrix=Sd, p=5)
    St_after_sparse = sparse_matrix(similarity_matrix=St, p=5)
    np.savetxt('Data/multi_similarity/drug_fusion_similarity_max_708_708_after_sparse.txt', Sd_after_sparse)
    np.savetxt('Data/multi_similarity/target_fusion_similarity_max_1512_1512_after_sparse.txt', St_after_sparse)
else:
    Sd_after_sparse = np.loadtxt('Data/multi_similarity/drug_fusion_similarity_max_708_708_after_sparse.txt')
    St_after_sparse = np.loadtxt('Data/multi_similarity/target_fusion_similarity_max_1512_1512_after_sparse.txt')

Ld = normalized_laplacian(Sd_after_sparse)
Lt = normalized_laplacian(St_after_sparse)

# 构建拉普拉斯矩阵 A = D（A+I）D
A_laplacians = laplacians(A)  # 此时的A矩阵已经包含了WNKN的消息


class AE(nn.Module):

    def __init__(self, num_features, hidden_size, num_classes):
        super(AE, self).__init__()
        self.enc_1 = GCNConv(num_features, hidden_size, add_self_loops=True)
        self.enc_2 = GCNConv(hidden_size, hidden_size, add_self_loops=True)
        self.enc_3 = GCNConv(hidden_size, hidden_size, add_self_loops=True)
        self.z_en_layer = GCNConv(hidden_size, num_classes, add_self_loops=True)

        self.dec_1 = GCNConv(num_classes, hidden_size, add_self_loops=True)
        self.dec_2 = GCNConv(hidden_size, hidden_size, add_self_loops=True)
        self.dec_3 = GCNConv(hidden_size, hidden_size, add_self_loops=True)
        self.x_de_layer = GCNConv(hidden_size, num_features, add_self_loops=True)

    def forward(self, x, edge_index):
        x, edge_index_ae = x, edge_index

        # 编码过程
        enc_h1 = torch.relu(self.enc_1(x, edge_index_ae))
        enc_hs = [enc_h1]

        enc_h2 = torch.relu(self.enc_2(enc_h1, edge_index_ae))
        enc_hs.append(enc_h2)

        enc_h3 = torch.relu(self.enc_3(enc_h2, edge_index_ae))
        enc_hs.append(enc_h3)

        z_en = self.z_en_layer(enc_h3, edge_index_ae)

        # 解码过程
        dec_h1 = torch.relu(self.dec_1(z_en, edge_index_ae))
        dec_h2 = torch.relu(self.dec_2(dec_h1, edge_index_ae))
        dec_h3 = torch.relu(self.dec_3(dec_h2, edge_index_ae))
        x_de = self.x_de_layer(dec_h3, edge_index_ae)

        return x_de, enc_hs, z_en


class DWR_GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class):  # 原始特征数 隐藏层 最终特征数
        # 两个图卷积层（gc1和gc2）
        super(DWR_GCN, self).__init__()
        # 初始化一个空字典用于存储GCNConv层
        self.gc = nn.ModuleDict()
        # 获取卷积层的层数
        self.layers = 3
        # autoencoder for intra information
        self.ae = AE(
            num_features=n_feat,
            hidden_size=n_hid,
            num_classes=n_class)

        # 输入(m+n)x(m+n) 输出(m+n)xl
        # 使用描述性字符串作为键
        self.gc['conv1'] = GCNConv(in_channels=n_feat, out_channels=n_hid, add_self_loops=True)
        self.gc['conv2'] = GCNConv(in_channels=n_hid, out_channels=n_class, add_self_loops=True)

    def forward(self, x, edge_index, edge_weight):
        # 重构
        x_de, enc_hs, z_en = self.ae(x, edge_index)

        # 第一层特征提取
        x_out = self.gc['conv1'](x, edge_index, edge_weight)
        x_out = torch.sigmoid(x_out)
        x_result = x_out
        x_all_layers = []  # 存储所有层最终输出的特征
        x_in = x_result

        # 残差连接
        for k in range(self.layers):
            # 计算图卷积操作
            x_out = torch.sparse.mm(A_laplacians, x_in)

            # 计算权重
            weights = (F.cosine_similarity(x_out, x_in, dim=1))
            x_result = torch.einsum('a,ab->ab', weights, x_out)  # 使用广播机制进行加权
            # 残差连接逻辑
            if k < len(enc_hs):  # 确保索引在有效范围内
                x_in = alpha * x_result + x_in
                # print('第', k, '层残差连接完成')
            else:
                x_in = x_result  # 如果超出范围，就只用 x_result
                print('超出范围')
            x_in = torch.relu(x_in)
            # print('第', k, '层特征提取完成')
            x_all_layers.append(x_in)

        # 特征聚合
        x_all_layers = torch.stack(x_all_layers, dim=1)
        x_all_layers = torch.mean(x_all_layers, dim=1)

        # 最后一层特征处理
        x_out = self.gc['conv2'](x_all_layers, edge_index, edge_weight)
        x_out = torch.tanh(x_out)

        # Dual Self-supervised Module
        # DWR-GCN O分布
        o = F.softmax(x_out, dim=1)
        # 自编码器Q分布
        q = F.softmax(z_en, dim=1)

        return x_out, x_de, z_en, q, o


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


# Discriminator 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(out_features, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.dis(x)
        return x


# 解码器
class Decoder(nn.Module):
    def forward(self, x):
        y = x.permute(1, 0)
        z = torch.mm(x, y)
        z = torch.Tensor.relu(z)
        return z


# Model and optimizer
decoder = Decoder()
D = Discriminator()
G = DWR_GCN(n_feat=in_features,
            n_hid=N_HID,
            n_class=out_features,
            )

loss_function_E = nn.MSELoss()  #
loss_function_G = nn.CrossEntropyLoss()
loss_function_D = nn.CrossEntropyLoss()

D_optimizer = torch.optim.Adam(D.parameters(), lr=LR * 0.1)
G_optimizer = torch.optim.Adam(G.parameters(), lr=LR)

A_laplacians = torch.from_numpy(A_laplacians).float().view(n + m, n + m)
X = torch.from_numpy(X).float().view(n + m, n + m)  # 此时的X矩阵已经有多源相似度融合的信息

decoder = decoder.to(device)
G = G.to(device)
D = D.to(device)
A_laplacians = A_laplacians.to(device)
X = X.to(device)


def loss_GRMF(Z, Ld, Lt, drug_num, lambda_l, lambda_d, lambda_t):
    A = Z[:drug_num]  # 前drug_num行代表药物的特征 A矩阵代表药物的特征
    B = Z[drug_num:]  # 后面代表靶点的特征 B矩阵代表靶点的特征
    A_loss = torch.norm(A)
    B_loss = torch.norm(B)
    A_loss = torch.square(A_loss)
    B_loss = torch.square(B_loss)
    loss_L2 = A_loss + B_loss
    Ld = torch.tensor(np.float32(Ld)).to(device)  # 将numpy转成tensor
    Lt = torch.tensor(np.float32(Lt)).to(device)

    loss_drug = torch.mm(torch.mm(A.t(), Ld), A).trace()
    loss_target = torch.mm(torch.mm(B.t(), Lt), B).trace()

    return lambda_l * loss_L2 + lambda_d * loss_drug + lambda_t * loss_target


# 开始训练
for epoch in range(EPOCH):
    Z, x_de, z_en, tem_q, tem_o = G(X, edge_index_A, edge_weight)

    G_loss0 = F.mse_loss(x_de, X).to(device)

    tem_q = tem_q.data
    p = target_distribution(tem_q)

    kl_loss1 = F.kl_div(tem_q.log(), p, reduction='batchmean')
    kl_loss2 = F.kl_div(tem_o.log(), p, reduction='batchmean')

    A_hat = decoder(Z)  # 解码
    G_loss1 = loss_function_E(A_hat, torch.from_numpy(A).float().view(n + m, n + m).to(device))
    # GRMF规范化
    G_loss2 = loss_GRMF(Z=Z, Ld=Ld, Lt=Lt, drug_num=n, lambda_l=lambda_l, lambda_d=lambda_d, lambda_t=lambda_t)
    G_loss = G_loss1 + G_loss2 + G_loss0 + beta * kl_loss1 + gamma * kl_loss2
    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()

    print('Epoch: ', epoch,
          '| train Encoding_loss: %.10f' % G_loss.item(),
          'G_loss0: %.10F' % G_loss0.item(),
          "G_loss1: %.10f" % G_loss1.item(),
          "G_loss2: %.10f" % G_loss2.item(),
          "kl_loss1: %.10f" % kl_loss1.item(),
          "kl_loss2: %.10f" % kl_loss2.item(), end=' ')

    Z = Z.data.cpu()  # 2220*200

    # CPU版本
    real_data = Variable(torch.randn(n + m, out_features)).to(device)  # 随机生成2220个200维，服从正太分布的向量
    fake_data = Variable(Z).to(device)
    real_label = Variable(torch.ones(n + m)).to(device)  # 为输入数据生成标签（真1假0）
    fake_label = Variable(torch.zeros(n + m)).to(device)

    # compute loss of real_data
    real_out = D(real_data)
    d_loss_real = loss_function_D(real_out, torch.Tensor.long(real_label).to(device))
    real_scores = real_out  # closer to 1 means better

    # compute loss of fake_data
    fake_out = D(fake_data)
    d_loss_fake = loss_function_D(fake_out, torch.Tensor.long(fake_label).to(device))
    fake_scores = fake_out  # closer to 0 means better

    # 更新判别器
    D_loss = d_loss_real + d_loss_fake
    print('| train Discriminator_loss: %.10f' % D_loss.item(), end=' ')
    D_optimizer.zero_grad()
    D_loss.backward()
    D_optimizer.step()

    # ===============train generator 生成器
    # compute loss of fake_img
    output = D(fake_data)
    G_loss = loss_function_G(output, torch.Tensor.long(real_label))
    print('| train Generator_loss: %.10f' % G_loss.item(), end='\n')

    # 更新生成器
    G_optimizer.zero_grad()
    G_loss.backward()
    G_optimizer.step()

result = A_hat.data.cpu().numpy()
embedding = Z.data.cpu().numpy()
# np.savetxt('training_result/score' + str(fold) + '.txt', result)
np.savetxt('training_result/embedding' + str(fold) + '.txt', embedding)
