#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import yaml
import configparser
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import numpy as np

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
sys.path.append("../")
from utilities import *
from model.l45_Normal_GCN import Customize_GCN
from model.l45_Novel_Node_GCN import Novel_Node_GCN
from model.l45_Novel_Edge_GCN import Novel_Edge_GCN
from model.l45_Novel_Node_CosSim_GCN import Novel_Node_GCN_Sim
from model.l45_Novel_Node_Att import Novel_Node_GAT
from model.l45_Novel_Node_Att_StructureN import Novel_Node_GAT_N_Sim
from model.l45_Novel_Node_GCN_balanced import Novel_Node_GCN_balanced
from model.l45_Novel_Node_Att_balanced import Novel_Node_GAT_balanced
from model.l45_Base_Att import Node_GAT
from model.activation_classifier import *


MODEL_DICT = {
    "customize": Customize_GCN,
    "novel_node": Novel_Node_GCN,
    'novel_edge': Novel_Edge_GCN,
    'novel_sim': Novel_Node_GCN_Sim,
    'novel_balanced': Novel_Node_GCN_balanced,
    'gat': Node_GAT,
    'novel_gat': Novel_Node_GAT,
    'gat_n_sim': Novel_Node_GAT_N_Sim,
    'novel_gat_balanced': Novel_Node_GAT_balanced,
}


def main(args):
    ##############################################################################
    # general parameters
    dataset_name_list = args.dataset_name

    for model_name in args.model_type:
        for dataset_name in dataset_name_list:

            model_type = MODEL_DICT[model_name]

            load_pretrained = args.load_pretrained

            aggr_list = args.aggr
            ##############################################################################
            # Show figures
            show_figures = args.show_figures

            # Get the task-specific hyper parameters for the data, model and training
            with open(args.hyperparam+dataset_name+'.yaml') as f:
                file = f.read()
                training_args = yaml.safe_load(file)

            hyper_args = training_args.get("hyper")
            dataset_args = training_args.get("data")
            model_args = training_args.get("model")
            seed_args = args.seeds


            # hyperparameters
            for seed in seed_args:
                print(f'Seed: {seed}')
                torch.random.manual_seed(seed)
                np.random.seed(int(seed))
                epochs = hyper_args.get("epochs")
                lr = hyper_args.get("lr")

                # data split
                train_test_split = dataset_args.get("train_test_split")

                # Model parameters
                num_hidden_units = model_args.get("num_hidden_units")
                num_classes = model_args.get("num_classes")
                num_layers = model_args.get("num_layers")

                # other parameters
                k = training_args.get("k")
                ##############################################################################

                G, labels = load_syn_data(dataset_name)
                if dataset_name in ['Twitch', 'Cora']:
                    data = prepare_real_data(G, train_test_split)
                else:
                    data = prepare_syn_data(G, labels, train_test_split)

                activation_list = ACTIVATION_LIST

                for aggr in aggr_list:

                    torch.random.manual_seed(seed)
                    print(f'Aggregator: {aggr}')

                    paths = prepare_output_paths(args, dataset_name, k, aggr, model_name, seed)

                    activation_to_clear = list(activation_list.keys())
                    for key in activation_to_clear:
                        activation_list.pop(key)

                    model = model_type(args, num_layers, data["x"].shape[1], num_hidden_units, num_classes, dataset_name, aggr)

                    if load_pretrained:
                        print("Loading pretrained model...")
                        model.load_state_dict(torch.load(os.path.join(paths['base'], f"model_{aggr}.pkl")))
                        model.eval()

                        with open(os.path.join(paths['base'], f"activations_{aggr}.txt"), 'rb') as file:
                            activation_list = pickle.loads(file.read())

                    else:
                        # model.apply(weights_init)
                        print('Training model...')
                        train(model, data, epochs, lr, paths['base'], show_figures)

                    x = data["x"]
                    edges = data["edges"]
                    y = data["y"]
                    train_mask = data["train_mask"]
                    test_mask = data["test_mask"]

                    model = model_type(args, num_layers, data["x"].shape[1], num_hidden_units, num_classes, dataset_name, aggr)
                    model.load_state_dict(torch.load(os.path.join(paths['base'], "model.pth")))

                    _ = test(model, x, y, edges, train_mask)

                    torch.random.manual_seed(seed)

                    # TSNE conversion
                    tsne_models = []
                    tsne_data = []
                    print('TSNE reduction...')
                    key = f'layers.{num_layers-1}'
                    layer_num = 0
                    activation = torch.squeeze(activation_list[key]).detach().numpy()
                    tsne_model = TSNE(n_components=2)
                    d = tsne_model.fit_transform(activation)
                    plot_activation_space(d, labels, "TSNE-Reduced", layer_num, paths['TSNE'], "(coloured by labels)",
                                          plot=show_figures)

                    tsne_models.append(tsne_model)
                    tsne_data.append(d)

                    # PCA conversion
                    pca_models = []
                    pca_data = []
                    print('PCA reduction...')
                    activation = torch.squeeze(activation_list[key]).detach().numpy()
                    pca_model = PCA(n_components=2)
                    d = pca_model.fit_transform(activation)
                    plot_activation_space(d, labels, "PCA-Reduced", layer_num, paths['PCA'], "(coloured by labels)",
                                          plot=show_figures)

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

                    plot_clusters(tsne_data[layer_num], pred_labels, "KMeans", k, layer_num, paths['KMeans'], "Raw", "_TSNE",
                                  "(TSNE Reduced)", plot=show_figures)
                    plot_clusters(pca_data[layer_num], pred_labels, "KMeans", k, layer_num, paths['KMeans'], "Raw", "_PCA",
                                  "(PCA Reduced)", plot=show_figures)
                    sample_graphs, sample_feat = plot_samples(kmeans_model, activation, data["y"], layer_num, k, "KMeans", "raw",
                                                              num_nodes_view, edges, num_expansions, paths['KMeans'],
                                                              plot=show_figures)
                    raw_sample_graphs.append(sample_graphs)
                    raw_kmeans_models.append(kmeans_model)

                    print('Training a decision tree...')
                    classifier_str = "decision_tree"

                    completeness_scores = []

                    i = 0
                    activation = torch.squeeze(activation_list[key]).detach().numpy()
                    activation_cls = ActivationClassifier(activation, raw_kmeans_models[i], classifier_str, data["x"], data["y"],
                                                          data["train_mask"], data["test_mask"])

                    d = ["Kmeans", "Raw", str(activation_cls.get_classifier_accuracy())]
                    completeness_scores.append(d)
                    print(d)

                    with open(paths['result'], "w") as f:
                        f.write(str(d))

                    print('---------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, nargs="*",
                        default=['BA_Shapes', 'BA_Community',
                                 'Tree_Cycle', 'Cora'],     # TODO: one more real dataset
                        choices=['BA_Shapes', 'BA_Grid', 'BA_Community',
                                 'Tree_Cycle', 'Tree_Grid', 'Twitch', 'Cora'])
    parser.add_argument('--model_type', type=str, nargs="*",
                        default=['novel_node'],
                        choices=['customize', 'novel_node', 'novel_edge',
                                 'novel_sim', 'gat', 'novel_gat', 'gat_n_sim', 'novel_balanced'])
    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--aggr', nargs="*",
                        default=['add', 'multi'],
                        choices=['add', 'mean', 'max', 'std', 'mul', 'div', 'min', 'multi'])
    # Pick one or two and use them on novel designed models
    parser.add_argument('--seeds', nargs="*",
                        default=[42])
    # Similarity Measure in Novel_Edge_GCN
    parser.add_argument('--similar_measure', type=str, default='edge',
                        choices=['edge', 'edit_dist'])
    # Similarity Measure in Novel_sim
    parser.add_argument('--node_similar_measure', type=str, default='cosine',
                        choices=['cosine', 'eu_dist'])
    # Similarity Measure in Novel_sim
    parser.add_argument('--norm_in_gat_n_sim', type=str, default='novel_node',
                        choices=['novel_node', '1', 'gcn'])
    # FIXME: max, std, mul don't work on my laptop...
    parser.add_argument('--hyperparam', type=str,
                        default="hyper_conf/")
    parser.add_argument('--show_figures', type=bool,
                        default=False)
    args = parser.parse_args()

    # assert args.dataset_name in args.hyperparam, \
    #     ValueError("Selected dataset and specified hyper-parameters should belong to the same dataset. ")

    set_rc_params()

    main(args)
