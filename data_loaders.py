from __future__ import print_function

import numpy as np
import random
import math
import os

import torch
import torchvision
import torch.utils.data
import re

def _base_dataset_name(name: str) -> str:
    """
    归一化数据集名：如 'p0.1_mnist' -> 'mnist', 'fashion-mnist' -> 'fmnist'
    """
    ds = name.lower().strip()
    if 'har' in ds:
        return 'har'
    if 'fashion' in ds or 'fmnist' in ds:
        return 'fmnist'
    if 'mnist' in ds:
        return 'mnist'
    return ds


def get_shapes(dataset):
    """
    Get the input and output shapes of the data examples for each dataset used.
    """
    ds = _base_dataset_name(dataset)

    if ds == 'har':
        num_inputs = 561
        num_outputs = 6
        num_labels = 6
    elif ds in ('mnist', 'fmnist'):
        # 图像输入交给模型处理；这里只给出占位与类别数
        num_inputs = 28 * 28
        num_outputs = 10
        num_labels = 10
    else:
        raise NotImplementedError
    return num_inputs, num_outputs, num_labels



def load_data(dataset, seed):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    ds = _base_dataset_name(dataset)

    if dataset == 'HAR':
        train_dir = os.path.join("data", "HAR", "train", "")
        test_dir = os.path.join("data", "HAR", "test", "")

        file = open(train_dir + "X_train.txt", 'r')
        X_train = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
        file.close()

        file = open(train_dir + "y_train.txt", 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_train = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        file = open(test_dir + "X_test.txt", 'r')
        X_test = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]], dtype=np.float32)
        file.close()

        file = open(test_dir + "y_test.txt", 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_test = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        # Loading which datapoint belongs to which client
        file = open(train_dir + "subject_train.txt", 'r')
        train_clients = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        file = open(test_dir + "subject_test.txt", 'r')
        test_clients = np.array([elem for elem in [row.replace('  ', ' ').strip().split(' ')[0] for row in file]], dtype=np.int32)
        file.close()

        X = np.concatenate((X_train, X_test))
        y = np.concatenate((y_train, y_test))

        y_train, y_test, X_train, X_test = [], [], [], []

        clients = np.concatenate((train_clients, test_clients))
        for client in range(1, 31):
            mask = tuple([clients == client])
            x_client = X[mask]
            y_client = y[mask]

            split = np.concatenate((np.ones(int(np.ceil(0.75*len(y_client))), dtype=bool), np.zeros(int(np.floor(0.25*len(y_client))), dtype=bool)))
            np.random.shuffle(split)  # Generate mask for train test split with ~0.75 1
            x_train_client = x_client[split]
            y_train_client = y_client[split]
            x_test_client = x_client[np.invert(split)]
            y_test_client = y_client[np.invert(split)]

            # Attach vector of client id to training data for data assignment in assign_data()
            x_train_client = np.insert(x_train_client, 0, client, axis=1)
            if len(X_train) == 0:
                X_train = x_train_client
                X_test = x_test_client
                y_test = y_test_client
                y_train = y_train_client
            else:
                X_train = np.append(X_train, x_train_client, axis=0)
                X_test = np.append(X_test, x_test_client, axis=0)
                y_test = np.append(y_test, y_test_client)
                y_train = np.append(y_train, y_train_client)

        tensor_train_X = torch.tensor(X_train, dtype=torch.float32)
        tensor_test_X = torch.tensor(X_test, dtype=torch.float32)
        tensor_train_y = torch.tensor(y_train, dtype=torch.int64) - 1
        tensor_test_y = torch.tensor(y_test, dtype=torch.int64) - 1
        train_dataset = torch.utils.data.TensorDataset(tensor_train_X, tensor_train_y)
        test_dataset = torch.utils.data.TensorDataset(tensor_test_X, tensor_test_y)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
        return train_loader, test_loader

    elif ds in ('mnist', 'fmnist'):
        import torchvision.transforms as T
        if ds == 'mnist':
            tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
            train_ds = torchvision.datasets.MNIST(root="./data", train=True,  download=True, transform=tfm)
            test_ds  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
        else:
            tfm = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
            train_ds = torchvision.datasets.FashionMNIST(root="./data", train=True,  download=True, transform=tfm)
            test_ds  = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=100, shuffle=True,
            worker_init_fn=seed_worker, generator=g
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=100, shuffle=False,
            worker_init_fn=seed_worker, generator=g
        )
        return train_loader, test_loader

    else:
        raise NotImplementedError


def assign_data(train_data, bias, device, num_labels=10, num_workers=100, server_pc=100, p=0.1, dataset="HAR", seed=1):
    """
    Assign the data to the clients.

    bias: [0,1]，越大越非IID。内部映射到 Dirichlet 的 alpha = max(1e-3, (1 - bias))
    server_pc/p: 服务器验证集大小与类别偏置，延续原有逻辑
    """
    ds = _base_dataset_name(dataset).upper()

    # 服务器样本类别配额（保留原逻辑）
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[1] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == 1:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])
    server_counter = [0 for _ in range(num_labels)]

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if ds == "HAR":
        # === 原 HAR 路线（保留） ===
        each_worker_data = [[] for _ in range(30)]
        each_worker_label = [[] for _ in range(30)]
        server_data, server_label = [], []
        for _, (data, label) in enumerate(train_data):
            data = data.to(device)
            label = label.to(device)
            for (x, y) in zip(data, label):
                clientId = int(x[0].item()) - 1
                x = x[1:len(x)].reshape(1, 561)
                if server_counter[int(y.cpu().numpy())] < samp_dis[int(y.cpu().numpy())]:
                    server_data.append(x); server_label.append(y)
                    server_counter[int(y.cpu().numpy())] += 1
                else:
                    each_worker_data[clientId].append(x)
                    each_worker_label[clientId].append(y)

        if server_pc != 0:
            server_data = torch.cat(server_data, dim=0)
            server_label = torch.stack(server_label, dim=0)
        else:
            server_data = torch.empty(size=(0, 561)).to(device)
            server_label = torch.empty(size=(0,)).to(device)

        each_worker_data = [torch.cat(w, dim=0) for w in each_worker_data]
        each_worker_label = [torch.stack(w, dim=0) for w in each_worker_label]

        # 随机打乱 30 个 HAR 客户
        random_order = np.random.RandomState(seed=seed).permutation(30)
        each_worker_data  = [each_worker_data[i]  for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
        return server_data, server_label, each_worker_data, each_worker_label

    elif ds in ("MNIST", "FMNIST"):
        # === 新：MNIST / FMNIST 路线（不依赖样本自带 clientId） ===
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]
        server_data, server_label = [], []

        # 收集各类样本
        by_class = [[] for _ in range(num_labels)]
        for _, (data, label) in enumerate(train_data):
            data = data.to(device)      # [B,1,28,28]
            label = label.to(device)    # [B]
            for (x, y) in zip(data, label):
                by_class[int(y.item())].append((x.unsqueeze(0), y))

        alpha = max(1e-3, (1.0 - bias))  # bias 越大，alpha 越小 => 越非IID
        for c in range(num_labels):
            pool = by_class[c]
            if not pool:
                continue
            rng.shuffle(pool)
            # Dirichlet 将该类别样本分到 num_workers 个客户端
            props = rng.dirichlet([alpha]*num_workers)
            counts = (props/props.sum() * len(pool)).astype(int)
            while counts.sum() < len(pool):
                counts[counts.argmax()] += 1

            s = 0
            for wid in range(num_workers):
                k = counts[wid]
                if k <= 0:
                    continue
                for (x, y) in pool[s:s+k]:
                    if server_counter[c] < samp_dis[c]:
                        server_data.append(x)   # [1,1,28,28]
                        server_label.append(y)
                        server_counter[c] += 1
                    else:
                        each_worker_data[wid].append(x)   # [1,1,28,28]
                        each_worker_label[wid].append(y)
                s += k

        if server_pc != 0 and len(server_data) > 0:
            server_data = torch.cat(server_data, dim=0)               # [Ns,1,28,28]
            server_label = torch.stack(server_label, dim=0)           # [Ns]
        else:
            server_data = torch.empty(size=(0, 1, 28, 28)).to(device)
            server_label = torch.empty(size=(0,)).to(device)

        # 合并每个客户端样本
        for wid in range(num_workers):
            if len(each_worker_data[wid]) > 0:
                each_worker_data[wid]  = torch.cat(each_worker_data[wid], dim=0)   # [Ni,1,28,28]
                each_worker_label[wid] = torch.stack(each_worker_label[wid], dim=0)
            else:
                each_worker_data[wid]  = torch.empty(size=(0, 1, 28, 28)).to(device)
                each_worker_label[wid] = torch.empty(size=(0,)).to(device)

        # 打乱客户端顺序（num_workers 个）
        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data  = [each_worker_data[i]  for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
        return server_data, server_label, each_worker_data, each_worker_label

    else:
        raise NotImplementedError
