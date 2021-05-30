import numpy as np
import torch

def euclidean_distance(input1, input2):
    m, n = input1.size(0), input2.size(0)
    distmat = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, input1, input2.t())
    distmat = distmat.clamp(min=1e-12).sqrt()
    return distmat

def load_train_features(path='resnetduke\train_features.npy'):
    train_features = np.load(path)
    train_features = torch.from_numpy(train_features)
    affinity = euclidean_distance(train_features, train_features)
    affinity = torch.exp(-1*affinity)
    return affinity

def load_test_features(g_path='resnetduke\gallary_features.npy', q_path='resnetduke\quary_features.npy'):
    g_features = np.load(g_path)
    q_features = np.load(q_path)
    g_features = torch.from_numpy(g_features)
    q_features = torch.from_numpy(q_features)
    test_features = torch.cat((q_features, g_features), 0)
    affinity = euclidean_distance(test_features, test_features)
    affinity = torch.exp(-1*affinity)
    return affinity

def load_train_label(path='resnetduke\train_pids.npy', use_gpu=True):
    pid = torch.from_numpy(np.load(path))
    if(use_gpu):
        pid = pid.cuda()
    train_label = build_label(pid)
    return train_label, pid

def load_test_label(g_path='resnetduke\gallary_pids.npy', q_path='resnetduke\quary_pids.npy', use_gpu=True):
    g_pid = torch.from_numpy(np.load(g_path))
    q_pid = torch.from_numpy(np.load(q_path))
    test_pid = torch.cat((q_pid, g_pid),0)
    test_label = build_label(test_pid)
    return test_label, test_pid

def build_label(pid):
    n = pid.size()[0]
    label = torch.zeros((n, n), dtype=float)
    for i in range(n):
        label[i] =  pid[i]==pid[:]
    return label

def init_edge(affinity, k):
    value, index = torch.sort(affinity, descending=False)
    index = index[:,0:k]
    value = value[:,0:k]
    n = index.size(0)
    edge_index = torch.arange(0,n)
    edge_index = edge_index.expand(k,n).transpose(0,1).contiguous()
    edge_index = edge_index.view(1, -1)
    value = value.contiguous().view(1,-1)
    index = index.contiguous().view(1,-1)
    edge_index = edge_index.to(index.device)
    edge_index = torch.cat((edge_index, index), 0)
    return edge_index, value

def init_edge_numpy(affinity, k):
    affinity = affinity.cpu().numpy()
    index = np.argsort(affinity, axis=1)
    index = torch.from_numpy(index)
    #value, index = torch.sort(affinity, descending=False)
    index = index[:,0:k]
    #value = value[:,0:k]
    n = index.size(0)
    edge_index = torch.arange(0,n)
    edge_index = edge_index.expand(k,n).transpose(0,1).contiguous()
    edge_index = edge_index.view(1, -1)
    #value = value.contiguous().view(1,-1)
    index = index.contiguous().view(1,-1)
    edge_index = edge_index.to(index.device)
    edge_index = torch.cat((edge_index, index), 0)
    return edge_index

def dijkstra(matrix, source):
    M = 1E100
    n = matrix.size(0)
    matrix = matrix.numpy()
    found = [source]        # 已找到最短路径的节点
    cost = [M] * n          # source到已找到最短路径的节点的最短距离
    cost[source] = 0
    while len(found) < n:   # 当已找到最短路径的节点小于n时
        min_value = M+1
        col = -1
        row = -1
        for f in found:     # 以已找到最短路径的节点所在行为搜索对象
            for i in [x for x in range(n) if x not in found]:   # 只搜索没找出最短路径的列
                if (matrix[f][i] + cost[f] < min_value) & (matrix[f][i]!=0):  # 找出最小值
                    min_value = matrix[f][i] + cost[f]  # 在某行找到最小值要加上source到该行的最短路径
                    row = f         # 记录所在行列
                    col = i
        if col == -1 or row == -1:  # 若没找出最小值且节点还未找完，说明图中存在不连通的节点
            break
        found.append(col)           # 在found中添加已找到的节点
        cost[col] = min_value       # source到该节点的最短距离即为min_value
    return  cost
    
