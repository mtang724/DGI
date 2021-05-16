import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from models import DGI, LogReg
from utils import process

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
from layers import mlp
from utils import process
from torch.utils.data import Dataset
import statistics

class NodeClassificationDataset(Dataset):
    def __init__(self, node_embeddings, labels):
        self.len = node_embeddings.shape[0]
        self.x_data = node_embeddings
        self.y_data = labels

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def read_roleid(path_to_file):
    role_id = []
    with open(path_to_file) as f:
        contents = f.readlines()
        for content in contents:
            role_id.append(float(content))
    return role_id

def cluster_graph(role_id, node_embeddings):
    colors = role_id
    nb_clust = len(np.unique(role_id))
    pca = PCA(n_components=5)
    trans_data = pca.fit_transform(StandardScaler().fit_transform(node_embeddings))
    km = KMeans(n_clusters=nb_clust)
    km.fit(trans_data)
    labels_pred = km.labels_

    ######## Params for plotting
    cmapx = plt.get_cmap('rainbow')
    x = np.linspace(0, 1, nb_clust + 1)
    col = [cmapx(xx) for xx in x]
    markers = {0: '*', 1: '.', 2: ',', 3: 'o', 4: 'v', 5: '^', 6: '<', 7: '>', 8: 3, 9: 'd', 10: '+', 11: 'x',
               12: 'D', 13: '|', 14: '_', 15: 4, 16: 0, 17: 1, 18: 2, 19: 6, 20: 7}

    for c in np.unique(role_id):
        indc = [i for i, x in enumerate(role_id) if x == c]
        plt.scatter(trans_data[indc, 0], trans_data[indc, 1],
                    c=np.array(col)[list(np.array(labels_pred)[indc])],
                    marker=markers[c % len(markers)], s=300)

    labels = role_id
    for label, c, x, y in zip(labels, labels_pred, trans_data[:, 0], trans_data[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
    return labels_pred, colors, trans_data, nb_clust


def unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust):
    ami = sk.metrics.adjusted_mutual_info_score(colors, labels_pred)
    sil = sk.metrics.silhouette_score(trans_data, labels_pred, metric='euclidean')
    ch = sk.metrics.calinski_harabasz_score(trans_data, labels_pred)
    hom = sk.metrics.homogeneity_score(colors, labels_pred)
    comp = sk.metrics.completeness_score(colors, labels_pred)
    #print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
    #print(str(hom) + '\t' + str(comp) + '\t' + str(ami) + '\t' + str(nb_clust) + '\t' + str(ch) + '\t' + str(sil))
    return hom, comp, ami, nb_clust, ch, sil

def draw_pca(role_id, node_embeddings):
    cmap = plt.get_cmap('hot')
    x_range = np.linspace(0, 0.8, len(np.unique(role_id)))
    coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_id))}
    node_color = [coloring[role_id[i]] for i in range(len(role_id))]
    pca = PCA(n_components=2)
    node_embedded = StandardScaler().fit_transform(node_embeddings)
    principalComponents = pca.fit_transform(node_embedded)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])
    principalDf['target'] = role_id
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 PCA Components', fontsize=20)
    targets = np.unique(role_id)
    for target in zip(targets):
        color = coloring[target[0]]
        indicesToKeep = principalDf['target'] == target
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1'],
                   principalDf.loc[indicesToKeep, 'principal component 2'],
                   s=50,
                   c=color)
    ax.legend(targets)
    ax.grid()
    plt.show()


def evaluate(model, embeddings, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(embeddings)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train_real_datasets(emb, node_labels):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(emb.shape)
    class_number = int(max(node_labels)) + 1
    input_dims = emb.shape
    FNN = mlp.MLP(num_layers=5, input_dim=input_dims[1], hidden_dim=input_dims[1] // 2, output_dim=class_number).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(FNN.parameters())
    dataset = NodeClassificationDataset(emb, node_labels)
    split = process.DataSplit(dataset, shuffle=True)
    train_loader, val_loader, test_loader = split.get_split(batch_size=64, num_workers=0)
    print(len(train_loader.dataset))
    print(len(node_labels), emb.shape)
    # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    best = float('inf')
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            print("here")
            # data = data.to(device)
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            y_pred = FNN(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correct = 0
                total = 0
                for data in val_loader:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = FNN(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    total += labels.size(0)
                    correct += torch.sum(predicted == labels)
            if loss < best:
                best = loss
                torch.save(FNN.state_dict(), 'best_mlp.pkl')
            print(str(epoch), correct / total)

    with torch.no_grad():
        FNN.load_state_dict(torch.load('best_mlp.pkl'))
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = FNN(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    print((correct / total).item())
    return (correct / total).item()

import sys
dataset_str = sys.argv[1]

def Average(lst):
    return sum(lst) / len(lst)
# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.0001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 512
sparse = True
nonlinearity = 'prelu'  # special name to separate parameters
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

if dataset_str == "cora" or dataset_str == "citeseer" or dataset_str == "pubmed":
    adj, features, labels, idx_train, idx_val, idx_test = process.load_real_data(dataset_str)
else:
    adj, features, labels = process.read_real_datasets(dataset_str)
homs, comps, amis, nb_clusts, chs, sils = [], [], [], [], [],[]
features, _ = process.preprocess_features(features)
acc = []
for i in range(5):
    # adj, features, node_ordering = process.load_synthetic_data(dataset + str(i))
    # adj, features, labels, idx_train, idx_val, idx_test = process.load_real_data(dataset_str)
    # adj, features, node_ordering = process.load_intro_data()

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    # hid_units = features.shape[1]
    # nb_classes = labels.shape[1]

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.FloatTensor(features[np.newaxis])
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
    # labels = torch.FloatTensor(labels[np.newaxis])
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    model = DGI(ft_size, hid_units, nonlinearity)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.to(device)
        features = features.to(device)
        if sparse:
            sp_adj = sp_adj.to(device)
        else:
            adj = adj.to(device)
        # labels = labels.to(device)
        # idx_train = idx_train.to(device)
        # idx_val = idx_val.to(device)
        # idx_test = idx_test.to(device)

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        if torch.cuda.is_available():
            shuf_fts = shuf_fts.to(device)
            lbl = lbl.to(device)

        logits = model(features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)

        loss = b_xent(logits, lbl)

        print('Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_dgi.pkl'))

    embeds0, embeds3, _, _, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
    embeddings = torch.squeeze(embeds3.cpu().detach())
    acc.append(train_real_datasets(embeddings, labels))

print("mean:")
print(statistics.mean(acc))
print("std:")
print(statistics.stdev(acc))
    # emb = torch.squeeze(embeds.detach(), dim=0)
    # acc = 0
    # for i in range(10):
    #     acc += train_real_datasets(emb, labels)
    # print (acc/10)
    # role_id = read_roleid("C://Users/Ming/Desktop/DGI/synthetic/np_{}{}.txt".format(dataset, i))
#     role_id = np.loadtxt("data/intro.out")
#     role_id = [role_id[i] for i in node_ordering]
#     # role_id = np.loadtxt("C://Users/Ming/Desktop/DGI/synthetic/np_{}{}.txt".format(dataset, i))
#     role_id_num = len(set(role_id))
#
#     embeddings = torch.squeeze(embeds3.cpu().detach())
#     print(embeddings.shape)
#     labels_pred, colors, trans_data, nb_clust = cluster_graph(role_id, embeddings)
#     draw_pca(role_id, embeddings)
#
#     hom, comp, ami, nb_clust, ch, sil = unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust)
#     homs.append(hom)
#     comps.append(comp)
#     sils.append(sil)
#     print(hom, comp, sil)
# print(Average(homs), Average(comps), Average(sils))

