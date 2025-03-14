import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace
import gc


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.server_model_param = {}
        self.client_model_params = {}
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.top_k = 10

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def fed_train_single_batch(self, model_client, batch_data, optimizers, user):
        """train a batch and return an updated model."""
        users, items, ratings = batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        reg_item_embedding_mlp = copy.deepcopy(self.server_model_param['embedding_item_mlp.weight'][user].data)
        optimizer, optimizer_u_mlp, optimizer_i_mlp, optimizer_i_gmf = optimizers
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            reg_item_embedding_mlp = reg_item_embedding_mlp.cuda()
        optimizer.zero_grad()
        optimizer_u_mlp.zero_grad()
        optimizer_i_mlp.zero_grad()
        optimizer_i_gmf.zero_grad()
        ratings_pred = model_client(items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        regularization_term_mlp = compute_regularization_mlp(model_client, reg_item_embedding_mlp)
        loss += (self.config['reg'] * regularization_term_mlp)
        loss.backward()
        optimizer.step()
        optimizer_u_mlp.step()
        optimizer_i_mlp.step()
        optimizer_i_gmf.step()

        del reg_item_embedding_mlp
        return model_client, loss.item()

    def aggregate_clients_params(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # construct the user relation graph via embedding similarity.
        if self.config['construct_graph_source'] == 'item':
            user_relation_graph_mlp = construct_user_relation_graph_via_item_mlp(round_user_params, self.config['num_items'],
                                                            self.config['latent_dim'],
                                                            self.config['similarity_metric'])
            user_relation_graph_gmf = construct_user_relation_graph_via_item_gmf(round_user_params, self.config['num_items'],
                                                            self.config['latent_dim_gmf'],
                                                            self.config['similarity_metric'])
        else:
            user_relation_graph = construct_user_relation_graph_via_user(round_user_params,
                                                            self.config['latent_dim'],
                                                            self.config['similarity_metric'])
        # select the top-k neighborhood for each user.
        topk_user_relation_graph_mlp = select_topk_neighboehood(user_relation_graph_mlp, self.config['neighborhood_size'],
                                                            self.config['neighborhood_threshold'])
        topk_user_relation_graph_gmf = select_topk_neighboehood(user_relation_graph_gmf, self.config['neighborhood_size'],
                                                            self.config['neighborhood_threshold'])
        # update item embedding via message passing.
        updated_item_embedding_mlp = MP_on_graph_mlp(round_user_params, self.config['num_items'], self.config['latent_dim'],
                                             topk_user_relation_graph_mlp, self.config['mp_layers'])
        updated_item_embedding_gmf = MP_on_graph_gmf(round_user_params, self.config['num_items'], self.config['latent_dim_gmf'],
                                                     topk_user_relation_graph_gmf, self.config['mp_layers'])
        del self.server_model_param['embedding_item_mlp.weight']
        del self.server_model_param['embedding_item_gmf.weight']
        self.server_model_param['embedding_item_mlp.weight'] = copy.deepcopy(updated_item_embedding_mlp)
        self.server_model_param['embedding_item_gmf.weight'] = copy.deepcopy(updated_item_embedding_gmf)
        del user_relation_graph_mlp
        del user_relation_graph_gmf
        del topk_user_relation_graph_mlp
        del topk_user_relation_graph_gmf
        del updated_item_embedding_mlp
        del updated_item_embedding_gmf
    


    def fed_train_a_round(self, all_train_data, round_id):
        """train a round."""
        # sample users participating in single round.
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
        # store users' model parameters of current round.
        round_participant_params = {}

        # initialize server parameters for the first round.
        if round_id == 0:
            self.server_model_param['embedding_item_mlp.weight'] = {}
            self.server_model_param['embedding_item_gmf.weight'] = {}
            for user in participants:
                self.server_model_param['embedding_item_mlp.weight'][user] = copy.deepcopy(self.model.state_dict()['embedding_item_mlp.weight'].data.cpu())
            self.server_model_param['embedding_item_mlp.weight']['global'] = copy.deepcopy(self.model.state_dict()['embedding_item_mlp.weight'].data.cpu())
            self.server_model_param['embedding_item_gmf.weight']['global'] = copy.deepcopy(self.model.state_dict()['embedding_item_gmf.weight'].data.cpu())
        # perform model updating for each participated user.
        for user in participants:
            # copy the client model architecture from self.model
            model_client = copy.deepcopy(self.model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated user embedding and aggregated item embedding from server
            # and use local updated mlp parameters from last round.
            if round_id != 0:
                # for participated users, load local updated parameters.
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if user in self.client_model_params.keys():
                    for key in self.client_model_params[user].keys():
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                user_param_dict['embedding_item_mlp.weight'] = copy.deepcopy(self.server_model_param['embedding_item_mlp.weight']['global'].data).cuda()
                user_param_dict['embedding_item_gmf.weight'] = copy.deepcopy(self.server_model_param['embedding_item_gmf.weight']['global'].data).cuda()
                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            # optimizer is responsible for updating mlp parameters.
            optimizer = torch.optim.SGD(
                [{"params": model_client.fc_layers.parameters()}, {"params": model_client.affine_output.parameters()}],
                lr=self.config['lr'])  # MLP optimizer
            # optimizer_u is responsible for updating user embedding.
            optimizer_u_mlp = torch.optim.SGD(model_client.embedding_user_mlp.parameters(),
                                          lr=self.config['lr'] / self.config['clients_sample_ratio'] * self.config[
                                              'lr_eta'] - self.config['lr'])  # User optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i_mlp = torch.optim.SGD(model_client.embedding_item_mlp.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'] -
                                             self.config['lr'])  # Item optimizer
            # optimizer_i is responsible for updating item embedding.
            optimizer_i_gmf = torch.optim.SGD(model_client.embedding_item_gmf.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'] -
                                             self.config['lr'])  # Item optimizer
            optimizers = [optimizer, optimizer_u_mlp, optimizer_i_mlp, optimizer_i_gmf]
            # load current user's training data and instance a train loader.
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    model_client, loss = self.fed_train_single_batch(model_client, batch, optimizers, user)
            # print('[User {}]'.format(user))
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' user embedding using a dict.
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            # round_participant_params[user] = copy.deepcopy(self.client_model_params[user])
            # del round_participant_params[user]['embedding_user.weight']
            round_participant_params[user] = {}
            round_participant_params[user]['embedding_item_mlp.weight'] = copy.deepcopy(self.client_model_params[user]['embedding_item_mlp.weight'])
            round_participant_params[user]['embedding_item_gmf.weight'] = copy.deepcopy(self.client_model_params[user]['embedding_item_gmf.weight'])
            #round_participant_params[user]['embedding_item.weight'] += Laplace(0, self.config['dp']).expand(round_participant_params[user]['embedding_item.weight'].shape).sample()
            del model_client
            torch.cuda.empty_cache()
            del user_train_data
            del user_dataloader
            del client_param
        # aggregate client models in server side.
        self.aggregate_clients_params(round_participant_params)
        del round_participant_params
        del participants
        gc.collect()


    def fed_evaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        # ratings for computing loss.
        temp = [0] * 100
        temp[0] = 1
        ratings = torch.FloatTensor(temp)
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            ratings = ratings.cuda()
        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None
        all_loss = {}
        for user in range(self.config['num_users']):
            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)
            user_param_dict = copy.deepcopy(self.model.state_dict())
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
            # user_param_dict['embedding_item.weight'] = copy.deepcopy(
            #     self.server_model_param['embedding_item.weight']['global'].data).cuda()
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                # obtain user's positive test information.
                test_user = test_users[user: user + 1]
                test_item = test_items[user: user + 1]
                # obtain user's negative test information.
                negative_user = negative_users[user * 99: (user + 1) * 99]
                negative_item = negative_items[user * 99: (user + 1) * 99]
                # perform model prediction.
                test_score = user_model(test_item)
                negative_score = user_model(negative_item)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
                ratings_pred = torch.cat((test_score, negative_score))
                loss = self.crit(ratings_pred.view(-1), ratings)
            all_loss[user] = loss.item()
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        return hit_ratio, ndcg, all_loss

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
