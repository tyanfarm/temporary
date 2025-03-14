import torch
from engine import Engine
from utils import use_cuda, resume_checkpoint


class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.latent_dim_gmf = config["latent_dim_gmf"]

        self.embedding_item_gmf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_gmf)

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=1, embedding_dim=self.latent_dim)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for in_size, out_size in zip(config['layers'][:-1], config['layers'][1:]):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim_gmf  + self.config['layers'][-1], out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        user_indices = torch.LongTensor([0 for _ in range(len(item_indices))]).cuda()
        
        user_embedding = self.embedding_user_mlp(user_indices)
        item_embedding = self.embedding_item_mlp(item_indices)
        mlp_vector = torch.cat([user_embedding, item_embedding], dim=-1)
        
        for layer in self.fc_layers:
            mlp_vector = torch.nn.ReLU()(layer(mlp_vector))
        
        item_embedding_gmf = self.embedding_item_gmf(item_indices)
        gmf_vector = item_embedding_gmf
        
        vector = torch.cat([mlp_vector, gmf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        
        return rating



class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        self.model = MLP(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)
