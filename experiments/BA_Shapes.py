#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
sys.path.append("../")
from utilities import *
from model.l45_BA_Shapes_GCN import *
from model.activation_classifier import *

set_rc_params()

# general parameters
dataset_name = "BA_Shapes"

model_type = Customize_BA_Shapes_GCN
load_pretrained = False

aggr_list = ['add', 'mean', 'max']

# hyperparameters
k = 10
torch.random.manual_seed(42)
# other parameters
train_test_split = 0.8
num_hidden_units = 20
num_classes = 4

epochs = 3000
lr = 0.001


paths = prepare_output_paths(dataset_name, k)

G, labels = load_syn_data(dataset_name)
data = prepare_syn_data(G, labels, train_test_split)

for aggr in aggr_list:
    torch.random.manual_seed(42)
    print(f'Aggregator: {aggr}')
    activation_to_clear = list(activation_list.keys())
    for key in activation_to_clear:
        activation_list.pop(key)

    model = model_type(data["x"].shape[1], num_hidden_units, num_classes, "BA-Houses", aggr)

    if load_pretrained:
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(os.path.join(paths['base'], f"model_{aggr}.pkl")))
        model.eval()

        with open(os.path.join(paths['base'], f"activations_{aggr}.txt"), 'rb') as file:
            activation_list = pickle.loads(file.read())

    else:
        #model.apply(weights_init)
        print('Training model...')
        train(model, data, epochs, lr, paths['base'])

    # TSNE conversion
    tsne_models = []
    tsne_data = []
    print('TSNE reduction...')
    key = 'conv3'
    layer_num = 0
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    tsne_model = TSNE(n_components=2)
    d = tsne_model.fit_transform(activation)
    plot_activation_space(d, labels, "TSNE-Reduced", layer_num, paths['TSNE'], "(coloured by labels)")

    tsne_models.append(tsne_model)
    tsne_data.append(d)


    # PCA conversion
    pca_models = []
    pca_data = []
    print('PCA reduction...')
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    pca_model = PCA(n_components=2)
    d = pca_model.fit_transform(activation)
    plot_activation_space(d, labels, "PCA-Reduced", layer_num, paths['PCA'], "(coloured by labels)")

    pca_models.append(pca_model)
    pca_data.append(d)


    print(f'KMEANS clustering with k={k}...')
    num_nodes_view = 5
    num_expansions = 1
    edges = data['edge_list'].numpy()

    raw_sample_graphs = []
    raw_kmeans_models = []

    activation = torch.squeeze(activation_list[key]).detach().numpy()
    kmeans_model = KMeans(n_clusters=k, random_state=0)
    kmeans_model = kmeans_model.fit(activation)
    pred_labels = kmeans_model.predict(activation)

    plot_clusters(tsne_data[layer_num], pred_labels, "KMeans", k, layer_num, paths['KMeans'], "Raw", "_TSNE", "(TSNE Reduced)")
    plot_clusters(pca_data[layer_num], pred_labels, "KMeans", k, layer_num, paths['KMeans'], "Raw", "_PCA", "(PCA Reduced)")
    sample_graphs, sample_feat = plot_samples(kmeans_model, activation, data["y"], layer_num, k, "KMeans", "raw", num_nodes_view, edges, num_expansions, paths['KMeans'])
    raw_sample_graphs.append(sample_graphs)
    raw_kmeans_models.append(kmeans_model)

    print('Training a decision tree...')
    classifier_str = "decision_tree"

    completeness_scores = []

    i = 0
    activation = torch.squeeze(activation_list[key]).detach().numpy()
    activation_cls = ActivationClassifier(activation, raw_kmeans_models[i], classifier_str, data["x"], data["y"], data["train_mask"], data["test_mask"])

    d = ["Kmeans", "Raw", str(activation_cls.get_classifier_accuracy())]
    completeness_scores.append(d)
    print(d)

    print('---------------------------')
