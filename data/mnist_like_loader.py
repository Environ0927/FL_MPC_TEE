import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np

def _tfm():
    return T.Compose([T.Grayscale(1), T.ToTensor(), T.Normalize((0.5,), (0.5,))])

def load_mnist(data_dir='./data'):
    tfm = _tfm()
    tr = torchvision.datasets.MNIST(root=data_dir, train=True,  download=True, transform=tfm)
    te = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=tfm)
    return tr, te

def load_fmnist(data_dir='./data'):
    tfm = _tfm()
    tr = torchvision.datasets.FashionMNIST(root=data_dir, train=True,  download=True, transform=tfm)
    te = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tfm)
    return tr, te

def dirichlet_split_indices(labels, n_clients=20, alpha=0.3, seed=42):
    rng = np.random.default_rng(seed)
    n_classes = int(np.max(labels)) + 1
    idx_by_cls = [np.where(labels == c)[0] for c in range(n_classes)]
    clients = [[] for _ in range(n_clients)]
    for c, idx in enumerate(idx_by_cls):
        idx = idx.copy(); rng.shuffle(idx)
        props = rng.dirichlet([alpha]*n_clients)
        counts = (props/props.sum()*len(idx)).astype(int)
        while counts.sum() < len(idx):
            counts[counts.argmax()] += 1
        s = 0
        for i in range(n_clients):
            k = counts[i]
            if k>0:
                clients[i].extend(idx[s:s+k].tolist())
                s += k
    return clients

def make_client_loaders(train_ds, client_indices, batch_size=64, shuffle=True):
    loaders = []
    for idxs in client_indices:
        loaders.append(DataLoader(Subset(train_ds, idxs), batch_size=batch_size, shuffle=shuffle))
    return loaders

def make_test_loader(test_ds, batch_size=256):
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False)
