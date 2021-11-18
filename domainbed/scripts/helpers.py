import torch
import numpy as np


class MyDataloader(torch.utils.data.Dataset):
    """
    Combine Seperated Datasets
    """
    def __init__(self, data_list):
        self.data = torch.utils.data.ConcatDataset(data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx


def get_features(data_loader, model, N, batch_size, device):
    """
    Input: dataloader, model used (eg: Resnet50), N: Size of dataset
    Returns: features, labels, idx
    """
    model.train(False)
    for i, ((data, labels), idx) in enumerate(data_loader):
        features = model.featurizer(data.to(device)).detach().cpu().numpy()  # get features
        features = np.asarray(features.reshape(features.shape[0], -1), dtype=np.float32)
        if i == 0:
            features_ = np.zeros((N, features.shape[1]), dtype=np.float32)
            labels_ = np.zeros(N, dtype=np.float32)
        if i < N - 1:
            features_[i * batch_size : (i + 1) * batch_size] = features
            labels_[i * batch_size : (i + 1) * batch_size] = labels
        else:
            features_[i * batch_size :] = features  # last batch
            labels_[i * batch_size :] = labels
    return features_, labels_


def get_cluster_labels(clustering, features):
    _, I = clustering.kmeans.index.search(features.copy(order="C"), 1)
    cluster_labels = [int(n[0]) for n in I]
    return cluster_labels


def get_images_list(num_clusters, len_data, cluster_labels):
    images_lists = [[] for i in range(num_clusters)]
    for i in range(len_data):
        images_lists[cluster_labels[i]].append(i)
    return images_lists


def get_hparam(hparams, hparams_seed):
    '''
    Function to Set hparam values
    '''
    hparam_num=0
    pca_offset=[8, 64, 128]
    num_clusters=[5]
    clust_epoch=[0]
    for p in pca_offset:
        for n in num_clusters:
            for e in clust_epoch:
                if hparams_seed == hparam_num:
                    hparams['num_clusters'] = n
                    hparams['offset'] = p
                    hparams['clust_epoch'] = e
                hparam_num+=1
    return hparams


def get_data_split_idx(dataset):
    l = 0
    idx_list = []
    for i in dataset:
        idx_list.append(list(range(l, len(i) + l)))
        l = len(i)+l  
    return idx_list
