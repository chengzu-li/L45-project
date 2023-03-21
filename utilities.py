import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import networkx as nx

from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.neighbors import NearestCentroid

from torch_geometric.datasets import Twitch, Planetoid

from tqdm import tqdm

def set_rc_params():
    """
    It sets some visulisation parameters
    """
    small = 14
    medium = 20
    large = 28

    plt.rc('figure', autolayout=True, figsize=(10, 6))
    plt.rc('font', size=medium)
    plt.rc('axes', titlesize=medium, labelsize=small, grid=True)
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)
    plt.rc('legend', fontsize=small)
    plt.rc('figure', titlesize=large, facecolor='white')
    plt.rc('legend', loc='upper left')

# global activation_list
# activation_list = {}
global ACTIVATION_LIST
ACTIVATION_LIST = {}

def get_activation(idx):
    '''Learned from: https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6'''
    def hook(model, input, output):
        ACTIVATION_LIST[idx] = output.detach()
    return hook

def register_hooks(model):
    """
    register hooks to extract activations
    :param model: the selected model we will extract activation from
    :return: the input model
    """
    for name, m in model.named_modules():
        m.register_forward_hook(get_activation(f"{name}"))

    return model

def weights_init(m):
    """
    Weight initialisation
    :param m: model we want to initialise
    """
    torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
    torch.nn.init.uniform_(m.bias.data)


def test(model, node_data_x, node_data_y, edge_list, mask):
    """
    Test phase of a model
    :param model: model we want to test
    :param node_data_x: node features
    :param node_data_y: labels
    :param edge_list: list of edges
    :param mask: mask
    :return: accuracy
    """
    # enter evaluation mode
    model.eval()

    correct = 0
    pred = model(node_data_x, edge_list).max(dim=1)[1]

    correct += pred[mask].eq(node_data_y[mask]).sum().item()
    return correct / (len(node_data_y[mask]))

def train(model, data, epochs, lr, path, plot=True):
    """
    Train the model
    :param model: the model we want to train
    :param data: dataset
    :param epochs: number of epochs
    :param lr: learning rate
    :param path: path where we want to save the model
    """
    # register hooks to track activation
    model = register_hooks(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # list of accuracies
    train_accuracies, test_accuracies, train_losses, test_losses = list(), list(), list(), list()

    # get data
    x = data["x"]
    edges = data["edges"]
    y = data["y"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]

    edge_index_0, edge_index_1 = (torch.cat((edges[0], edges[1])).unsqueeze(0),
                                  torch.cat((edges[1], edges[0])).unsqueeze(0))
    edges = torch.cat((edge_index_0, edge_index_1), dim=0)

    # iterate for number of epochs
    torch.autograd.set_detect_anomaly(True)
    best_loss = None
    strike = 0
    for epoch in tqdm(range(epochs)):
            # set mode to training
            model.train()
            optimizer.zero_grad()

            # input data
            out = model(x, edges)

            # calculate loss
            loss = F.nll_loss(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                test_loss = F.nll_loss(out[test_mask], y[test_mask])

                # get accuracy
                train_acc = test(model, x, y, edges, train_mask)
                test_acc = test(model, x, y, edges, test_mask)

            if epoch % 20 == 0:
                if best_loss == None or best_loss >= test_loss:
                    strike = 0
                    best_loss = test_loss
                    best_train_acc = train_acc
                    best_test_acc = test_acc
                    print('Saving models...')
                    torch.save(model.state_dict(), os.path.join(path, "model.pth"))
                else:
                    strike += 1
                    if strike == 7:
                        break

            ## add to list and print
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())

            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
                  format(epoch, loss.item(), train_acc, test_acc), end = "\r")

            # if train_acc >= 0.95 and test_acc >= 0.95:
            #     break
    # plut accuracy graph
    acc_fig = plt.figure()
    accuracy_fig = acc_fig.add_subplot()
    accuracy_fig.plot(train_accuracies, label="Train Accuracy")
    accuracy_fig.plot(test_accuracies, label="Testing Accuracy")
    accuracy_fig.set_title(f"Accuracy of {model.name} Model during Training")
    accuracy_fig.set_xlabel("Epoch")
    accuracy_fig.set_ylabel("Accuracy")
    accuracy_fig.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_accuracy_plot.png"))
    if plot:
        plt.show()

    # loss figure
    fig = plt.figure()
    loss_fig = fig.add_subplot()
    loss_fig.plot(train_losses, label="Train Loss")
    loss_fig.plot(test_losses, label="Testing Loss")
    loss_fig.set_title(f"Loss of {model.name} Model during Training")
    loss_fig.set_xlabel("Epoch")
    loss_fig.set_ylabel("Loss")
    loss_fig.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_loss_plot.png"))
    if plot:
        plt.show()

    # save model
    # torch.save(model.state_dict(), os.path.join(path, "model.pkl"))

    # with open(os.path.join(path, "activations.txt"), 'wb') as file:
    #     pickle.dump(ACTIVATION_LIST, file)

    with open(os.path.join(path, "accuracy.txt"), 'wb') as file:
        result = [f"train_acc: {best_train_acc}", f"test_acc: {best_test_acc}"]
        pickle.dump(str(result), file)

def prepare_output_paths(args, dataset_name, k, aggr, model_name, seed):
    """
    Setup paths
    :param dataset_name: name of the dataset we are using
    :param k: number of cluster
    :return: dict with different path that you can use
    """
    if model_name != "customized":
        path = f"output-{model_name}/{dataset_name}/{aggr}/{seed}"
        if model_name == 'novel_edge' and args.similar_measure == "edit_dist":
            path = f"output-{model_name}-edit_dist/{dataset_name}/{aggr}/{seed}"
        elif model_name == 'novel_sim' and args.node_similar_measure == 'eu_dist':
            path = f"output-{model_name}-eu_dist/{dataset_name}/{aggr}/{seed}"
        elif model_name == 'gat_n_sim' and args.norm_in_gat_n_sim != "novel_node":
            path = f"output-{model_name}-{args.norm_in_gat_n_sim}/{dataset_name}/{aggr}/{seed}"
    else:
        path = f"output/{dataset_name}/{aggr}/{seed}"
    path_tsne = os.path.join(path, "TSNE")
    path_pca = os.path.join(path, "PCA")
    path_umap = os.path.join(path, "UMAP")
    path_kmeans = os.path.join(path, f"{k}_KMeans")
    path_hc = os.path.join(path, f"HC")
    path_ward = os.path.join(path, f"WARD")
    path_dbscan = os.path.join(path, f"DBSCAN")
    path_edges = os.path.join(path, f"edges")
    path_final_result = os.path.join(path, f"results.txt")
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_tsne, exist_ok=True)
    os.makedirs(path_pca, exist_ok=True)
    os.makedirs(path_umap, exist_ok=True)
    os.makedirs(path_kmeans, exist_ok=True)
    os.makedirs(path_hc, exist_ok=True)
    os.makedirs(path_ward, exist_ok=True)
    os.makedirs(path_dbscan, exist_ok=True)
    os.makedirs(path_edges, exist_ok=True)

    return {"base": path, "TSNE": path_tsne, "PCA": path_pca, "UMAP": path_umap, "KMeans": path_kmeans, "HC": path_hc,
            "Ward": path_ward, "DBSCAN": path_dbscan, "edges": path_edges, "result": path_final_result}

def load_syn_data(dataset_str):
    """
    Load a dataset
    :param dataset_str: name of the dataset
    :return: graph, labels
    """
    if dataset_str == "BA_Shapes":
        G = nx.readwrite.read_gpickle("../data/BA_Houses/graph_ba_300_80.gpickel")
        role_ids = np.load("../data/BA_Houses/role_ids_ba_300_80.npy")

    elif dataset_str == "BA_Grid":
        G = nx.readwrite.read_gpickle("../data/BA_Grid/graph_ba_300_80.gpickel")
        role_ids = np.load("../data/BA_Grid/role_ids_ba_300_80.npy")

    elif dataset_str == "BA_Community":
        G = nx.readwrite.read_gpickle("../data/BA_Community/graph_ba_350_100_2comm.gpickel")
        role_ids = np.load("../data/BA_Community/role_ids_ba_350_100_2comm.npy")

    elif dataset_str == "Tree_Cycle":
        G = nx.readwrite.read_gpickle("../data/Tree_Cycle/graph_tree_8_60.gpickel")
        role_ids = np.load("../data/Tree_Cycle/role_ids_tree_8_60.npy")

    elif dataset_str == "Tree_Grid":
        G = nx.readwrite.read_gpickle("../data/Tree_Grid/graph_tree_8_80.gpickel")
        role_ids = np.load("../data/Tree_Grid/role_ids_tree_8_80.npy")

    elif dataset_str == 'Twitch':
        G = Twitch('../data/Twitch', 'EN')[0]
        role_ids = G.y

    elif dataset_str == 'Cora':
        G = Planetoid('../data/Cora', 'cora')[0]
        role_ids = G.y

    else:
        raise Exception("Invalid Syn Dataset Name")

    return G, role_ids

def prepare_real_data(G, train_split):
    train_mask = np.random.rand(len(G.x)) < train_split
    test_mask = ~train_mask
    edge_list = torch.transpose(G.edge_index, 1, 0)
    data = {"x": G.x, "y": G.y, "edges": G.edge_index, "edge_list": edge_list, "train_mask": train_mask,
         "test_mask": test_mask}
    return data

def prepare_syn_data(G, labels, train_split, if_adj=False):
    """
    Preprocessing of dataset
    :param G: graph
    :param labels: labels
    :param train_split: ratio of train-test
    :param if_adj: True if we want to return the adjacency matrix
    :return: dict which contains "x":feature, "y":labels, "edges":edges rapresented with couple
            "edge_list":edges in a sparse matrix fashion, "train_mask":train mask, "test_mask":test_mask
    """
    existing_node = list(G.nodes)[-1]
    feat_dim = G.nodes[existing_node]["feat"].size
    features = np.zeros((G.number_of_nodes(), feat_dim), dtype=float)
    for i, u in enumerate(G.nodes()):
        features[i, :] = G.nodes[u]["feat"]

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels)

    edge_list = torch.from_numpy(np.array(G.edges))

    edges = edge_list.transpose(0, 1).long()
    if if_adj:
        edges = torch.tensor(nx.to_numpy_matrix(G), requires_grad=True, dtype=torch.float)

    train_mask = np.random.rand(len(features)) < train_split
    test_mask = ~train_mask

    print("Task: Node Classification")
    print("Number of features: ", len(features))
    print("Number of labels: ", len(labels))
    print("Number of classes: ", len(set(labels)))
    print("Number of edges: ", len(edges))

    return {"x": features, "y": labels, "edges": edges, "edge_list": edge_list, "train_mask": train_mask, "test_mask": test_mask}

def plot_activation_space(data, labels, activation_type, layer_num, path, note="", naming_help="", plot=True):
    """
    Plot the activation space after a dimentionality reduction
    :param data: features
    :param labels: labels
    :param activation_type: (str) kind of dimentionality reduction
    :param layer_num: number of the layer
    :param path: path where you want to save the plot
    :param note: extra info
    :param naming_help: name of the layer
    """
    rows = len(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"{activation_type} Activations of Layer {layer_num} {note}")

    scatter = ax.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow')
    ax.legend(handles=scatter.legend_elements()[0], labels=list(np.unique(labels)), bbox_to_anchor=(1.05, 1))

    plt.savefig(os.path.join(path, f"{layer_num}_layer{naming_help}.png"))
    if plot:
        plt.show()

def plot_clusters(data, labels, clustering_type, k, layer_num, path, data_type, reduction_type="", note="", plot=True):
    """
    Plot cluster
    :param data: features
    :param labels: labels
    :param clustering_type: type of clustering used
    :param k: number of clusters
    :param layer_num: number of the layer
    :param path: path where we want to save the image
    :param data_type: Type of data
    :param reduction_type: Dimentionality reduction used
    :param note: extra info
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'{clustering_type} Clustered {data_type} Activations of Layer {layer_num} {note}')

    for i in range(k):
        scatter = ax.scatter(data[labels == i,0], data[labels == i,1], label=f'Cluster {i}')

    ncol = 1
    if k > 20:
        ncol = int(k / 20) + 1
    ax.legend(bbox_to_anchor=(1.05, 1), ncol=ncol)
    plt.savefig(os.path.join(path, f"{layer_num}layer_{data_type}{reduction_type}.png"))
    if plot:
        plt.show()

def get_node_distances(clustering_model, data):
    """
    Get the list of node in a cluster sorted by their distance from the centroid of the cluster
    :param clustering_model: the trained clustering model
    :param data: feature and labels
    :return: list of node in a cluster sorted by their distance from the centroid of the cluster
    """
    if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
        x, y_predict = data
        clf = NearestCentroid()
        clf.fit(x, y_predict)
        centroids = clf.centroids_
        res = pairwise_distances(centroids, x)
        res_sorted = np.argsort(res, axis=-1)
    elif isinstance(clustering_model, KMeans):
        res_sorted = clustering_model.transform(data)

    return res_sorted

def plot_samples(clustering_model, data, y, layer_num, k, clustering_type, reduction_type, num_nodes_view, edges,
                 num_expansions, path, graph_data=None, graph_name=None, plot=True):
    """
    Plot the num_node_view nearest to the centroid of the cluster, for each cluster
    :param clustering_model: the trained clustering model
    :param data: feature
    :param y: labels
    :param layer_num: number of the layer
    :param k: number of cluster
    :param clustering_type: type of clustering
    :param reduction_type: type of dementionality reduction
    :param num_nodes_view: num of samples to show for each cluster
    :param edges: edges
    :param num_expansions: how many hop to show
    :param path: path where we want to save
    :return: subgraphs and features
    """
    res_sorted = get_node_distances(clustering_model, data)

    if isinstance(num_nodes_view, int):
        num_nodes_view = [num_nodes_view]
    col = sum([abs(number) for number in num_nodes_view])

    fig, axes = plt.subplots(k, col, figsize=(18, 3 * k + 2))
    fig.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num}', y=1.005)

    if graph_data is not None:
        fig2, axes2 = plt.subplots(k, col, figsize=(18, 3 * k + 2))
        fig2.suptitle(f'Nearest Instances to {clustering_type} Cluster Centroid for {reduction_type} Activations of Layer {layer_num} (by node index)', y=1.005)

    l = list(range(0, k))
    sample_graphs = []
    sample_feat = []

    for i, ax_list in zip(l, axes):
        if isinstance(clustering_model, AgglomerativeClustering) or isinstance(clustering_model, DBSCAN):
            distances = res_sorted[i]
        elif isinstance(clustering_model, KMeans):
            distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            tg, cm, labels, node_labels = get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data, graph_name)
            top_graphs = top_graphs + tg
            color_maps = color_maps + cm

        if graph_data is None:
            for ax, new_G, color_map, g_label in zip(ax_list, top_graphs, color_maps, labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)
        else:
            for ax, new_G, color_map, g_label, n_labels in zip(ax_list, top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax, labels=n_labels)
                ax.set_title(f"label {g_label}", fontsize=14)

            for ax, new_G, color_map, g_label, n_labels in zip(axes2[i], top_graphs, color_maps, labels, node_labels):
                nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
                ax.set_title(f"label {g_label}", fontsize=14)

        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])

    views = ''.join((str(i) + "_") for i in num_nodes_view)
    if isinstance(clustering_model, AgglomerativeClustering):
        fig.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view.png"))
    else:
        fig.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_{views}view.png"))

    if graph_data is not None:
        if isinstance(clustering_model, AgglomerativeClustering):
            fig2.savefig(os.path.join(path, f"{k}h_{layer_num}layer_{clustering_type}_{views}view_by_node.png"))
        else:
            fig2.savefig(os.path.join(path, f"{layer_num}layer_{clustering_type}_{reduction_type}_{views}view_by_node.png"))
    if plot:
        plt.show()

    return sample_graphs, sample_feat

def get_top_subgraphs(top_indices, y, edges, num_expansions, graph_data=None, graph_name=None):
    """
    Retreives the subgraph of the top_indices nodes
    :param top_indices: nodes
    :param y: labels
    :param edges: edges
    :param num_expansions: hop to show
    :return: subgraphs, color to show, labels of the subgraph, node labels
    """
    graphs = []
    color_maps = []
    labels = []
    node_labels = []

    df = pd.DataFrame(edges)

    for idx in top_indices:
        # get neighbours
        neighbours = list()
        neighbours.append(idx)

        for i in range(0, num_expansions):
            new_neighbours = list()
            for e in edges:
                if (e[0] in neighbours) or (e[1] in neighbours):
                    new_neighbours.append(e[0])
                    new_neighbours.append(e[1])

            neighbours = neighbours + new_neighbours
            neighbours = list(set(neighbours))

        new_G = nx.Graph()
        df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
        remaining_edges = df_neighbours.to_numpy()
        new_G.add_edges_from(remaining_edges)

        color_map = []
        node_label = {}
        if graph_data is None:
            for node in new_G:
                if node in top_indices:
                    color_map.append('green')
                else:
                    color_map.append('pink')
        else:
            if graph_name == "Mutagenicity":
                ids = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]
            elif graph_name == "REDDIT-BINARY":
                ids = []

            for node in zip(new_G):
                node = node[0]
                color_idx = graph_data[node]
                color_map.append(color_idx)
                node_label[node] = f"{ids[color_idx]}"

        color_maps.append(color_map)
        graphs.append(new_G)
        labels.append(y[idx])
        node_labels.append(node_label)


    return graphs, color_maps, labels, node_labels