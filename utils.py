"""
    Some handy functions for pytroch model training ...
"""
import torch
import numpy as np
import copy
from sklearn.metrics import pairwise_distances
import logging
import math
import os


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                     lr=params['lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer


def construct_user_relation_graph_via_item_mlp(round_user_params, item_num, latent_dim, similarity_metric):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num * latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item_mlp.weight'].numpy().flatten()
    # construct the user relation graph.
    adj = pairwise_distances(item_embedding, metric=similarity_metric)
    del item_embedding
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj
    
def construct_user_relation_graph_via_item_gmf(round_user_params, item_num, latent_dim, similarity_metric):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num * latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item_gmf.weight'].numpy().flatten()
    # construct the user relation graph.
    adj = pairwise_distances(item_embedding, metric=similarity_metric)
    del item_embedding
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj


def construct_user_relation_graph_via_user(round_user_params, latent_dim, similarity_metric):
    # prepare the user embedding array.
    user_embedding = np.zeros((len(round_user_params), latent_dim), dtype='float32')
    for user in round_user_params.keys():
        user_embedding[user] = copy.deepcopy(round_user_params[user]['embedding_user.weight'].numpy())
    # construct the user relation graph.
    adj = pairwise_distances(user_embedding, metric=similarity_metric)
    del user_embedding
    if similarity_metric == 'cosine':
        return adj
    else:
        return -adj


def select_topk_neighboehood(user_realtion_graph, neighborhood_size, neighborhood_threshold):
    topk_user_relation_graph = np.zeros(user_realtion_graph.shape, dtype='float32')
    if neighborhood_size > 0:
        for user in range(user_realtion_graph.shape[0]):
            user_neighborhood = user_realtion_graph[user]
            topk_indexes = user_neighborhood.argsort()[-neighborhood_size:][::-1]
            for i in topk_indexes:
                topk_user_relation_graph[user][i] = 1/neighborhood_size
    else:
        similarity_threshold = np.mean(user_realtion_graph)*neighborhood_threshold
        for i in range(user_realtion_graph.shape[0]):
            high_num = np.sum(user_realtion_graph[i] > similarity_threshold)
            if high_num > 0:
                for j in range(user_realtion_graph.shape[1]):
                    if user_realtion_graph[i][j] > similarity_threshold:
                        # neighbor_of_j = np.sum(user_realtion_graph[j] > similarity_threshold);
                        # User cang tuong dong voi nhieu nguoi thi dong gop cang lon 
                        topk_user_relation_graph[i][j] =  1 / high_num
            else:
                topk_user_relation_graph[i][i] = 1

    return topk_user_relation_graph


def MP_on_graph_mlp(round_user_params, item_num, latent_dim, topk_user_relation_graph, layers):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num*latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item_mlp.weight'].numpy().flatten()

    # aggregate item embedding via message passing.
    layers_result = [item_embedding];
    aggregated_item_embedding = np.matmul(topk_user_relation_graph, item_embedding);
    layers_result.append(aggregated_item_embedding);

    for layer in range(layers-1):
        result = np.matmul(topk_user_relation_graph, layers_result[-1])
        layers_result.append(result);

    layers_result = np.stack(layers_result, axis = 0)
    aggregated_item_embedding = 1/(1 + layers) * np.sum(layers_result, axis = 0)
    
    # reconstruct item embedding.
    item_embedding_dict = {}
    for user in round_user_params.keys():
        item_embedding_dict[user] = torch.from_numpy(aggregated_item_embedding[user].reshape(item_num, latent_dim))
    item_embedding_dict['global'] = sum(item_embedding_dict.values())/len(round_user_params)
    del item_embedding
    del layers_result
    del aggregated_item_embedding

    return item_embedding_dict

def MP_on_graph_gmf(round_user_params, item_num, latent_dim, topk_user_relation_graph, layers):
    # prepare the item embedding array.
    item_embedding = np.zeros((len(round_user_params), item_num*latent_dim), dtype='float32')
    for user in round_user_params.keys():
        item_embedding[user] = round_user_params[user]['embedding_item_gmf.weight'].numpy().flatten()

    layers_result = [item_embedding]

    for layer in range(layers):
        aggregated_item_embedding = np.matmul(topk_user_relation_graph, layers_result[-1])
        layers_result.append(aggregated_item_embedding)

    # layers_result = np.stack(layers_result, axis = 0)
    # aggregated_item_embedding = 1/(1 + layers) * np.sum(layers_result, axis = 0)
    # print(aggregated_item_embedding.shape)
    # print(item_embedding.shape)
    # print(topk_user_relation_graph.shape)
    
    item_embedding_dict = {}
    # for user in round_user_params.keys():
    #     item_embedding_dict[user] = torch.from_numpy(aggregated_item_embedding[user].reshape(item_num, latent_dim))
    aggregated_item_embedding = np.mean(aggregated_item_embedding, axis = 0)
    item_embedding_dict['global'] = torch.from_numpy(aggregated_item_embedding.reshape(item_num, latent_dim))
    del item_embedding
    del aggregated_item_embedding
    
    for i in layers_result:
        del i

    del layers_result
    return item_embedding_dict


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def compute_regularization_mlp(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction='mean')
    for name, param in model.named_parameters():
        if name == "embedding_item_mlp.weight":
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss
        
def compute_regularization_gmf(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction="mean")
    for name, param in model.named_parameters():
        if name == "embedding_item_gmf.weight":
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss
        
def save_weights_embedding(item_weights, save_dir="weights"):
    os.makedirs(save_dir, exist_ok=True)

    # Save .pt file (by torch)
    # Save weights item
    torch.save(item_weights, f"{save_dir}/item_weights.pt")

    print(f"Weights have been saved to {save_dir} folder")

    # Save .txt file
    with open(f"{save_dir}/training_logs.txt", "w") as log_file:
        log_file.write("Item Weights Saved:\n")
        for user, weight_dict in item_weights.items():
            for key, tensor in weight_dict.items():  # weight_dict is dict contains tensor
                log_file.write(f"User {user}, {key}: {tensor.numpy()}\n")

    print(f"Weights format text have been saved to training_logs.txt")