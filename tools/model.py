import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tools.dataprocess import *
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv, GINConv, GATConv, ChebConv, GAE, global_mean_pool, global_max_pool


# class GraphEncoder(nn.Module):
#     def __init__(self, input_dim: int, output_dim: int, dropout=0.2):
#         super(GraphEncoder, self).__init__()
#         self.conv1 = GATConv(input_dim, input_dim, heads=10)
#         self.conv2 = GCNConv(input_dim*10, input_dim*10)
#         self.fc_g1 = torch.nn.Linear(input_dim*10, 256)
#         self.fc_g2 = torch.nn.Linear(256, output_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, data: DATA.data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = self.relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.relu(x)
#         x = self.fc_g1(x)
#         x = self.dropout(x)
#         x = self.fc_g2(x)
#         return x



class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        # self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        # self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        # self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv1 = ChebConv(in_channels, 2 * out_channels, K=2)
        self.bn1 = nn.BatchNorm1d(2 * out_channels)
        self.conv_mu = ChebConv(2 * out_channels, out_channels, K=2)
        self.conv_logstd = ChebConv(2 * out_channels, out_channels, K=2)
        self.relu = nn.ReLU()
    def forward(self, data: DATA.data, batch=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        mu, logstd = self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        # mu = global_mean_pool(mu, batch=batch)
        # logstd = global_mean_pool(logstd, batch=batch)
        return mu, logstd


class gEncoder(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, dropout=0.2):
        super(gEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.BatchNorm1d(input_dim//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class gDecoder(nn.Module):
    def __init__(self, recon_dim:int, emb_dim:int, dropout=0.2):
        super(gDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2),
            nn.BatchNorm1d(emb_dim*2),
            nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(emb_dim*2, recon_dim)
        )
    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout=0.2):
        super(GraphEncoder, self).__init__()
        # self.conv1 = GATConv(input_dim, input_dim, heads=10)
        # self.conv2 = GCNConv(input_dim*10, input_dim*10)
        # self.fc_g1 = torch.nn.Linear(input_dim*10, 256)
        # self.fc_g2 = torch.nn.Linear(256, output_dim)
        self.conv1 = ChebConv(in_channels=input_dim, out_channels=128, K=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = ChebConv(in_channels=128, out_channels=128, K=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc_g1 = torch.nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc_g2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: DATA.data, batch=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc_g1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x_mean = global_mean_pool(x, batch=batch)
        # x_mean = global_max_pool(x, batch=batch)
        return x, x_mean


class GraphDecoder(nn.Module):
    def __init__(self, recon_dim: int, emb_dim: int, dropout=0.2):
        super(GraphDecoder, self).__init__()
        self.fc_g1 = torch.nn.Linear(emb_dim, 1024)
        self.fc_g2 = torch.nn.Linear(1024, recon_dim)
        # self.conv1 = GCNConv(recon_dim, recon_dim)
        self.conv1 = ChebConv(recon_dim, recon_dim, K=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index:torch.Tensor):
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.fc_g2(x)
        x = self.relu(x)
        x = self.conv1(x, edge_index)
        # x = self.relu(x)
        # x = self.conv1(x)
        return x


class Edgeindexdecoder(nn.Module):
    def __init__(self, input_dim:int):
        super(Edgeindexdecoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.fc2 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, edge_index:torch.tensor):
        edge_index = self.fc1(edge_index)
        edge_index = self.relu(edge_index)
        edge_index = self.drop(edge_index)
        edge_index = self.fc2(edge_index)
        return edge_index



class Classify(nn.Module):
    def __init__(self, input_dim):
        super(Classify, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return (self.net(x)).view(-1)


class Classify_savefeature(nn.Module):
    def __init__(self, input_dim):
        super(Classify_savefeature, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x), (self.net(x)).view(-1)
    

class Classify_savefeature1(nn.Module):
    def __init__(self, input_dim):
        super(Classify_savefeature1, self).__init__()

        self.l1 = nn.Linear(input_dim, 10)
        self.b1 = nn.BatchNorm1d(10)
        self.r = nn.ReLU()
        self.l2 = nn.Linear(10, 1)


    def forward(self, x):
        out1 = self.r(self.b1(self.l1(x)))
        out2 = self.l2(out1)
        return out1, out2.view(-1)


class Classifydim2(nn.Module):
    def __init__(self, input_dim):
        super(Classifydim2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 2)
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class projector(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(projector, self).__init__()
        self.l1 = nn.Linear(in_dim, in_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.re = nn.ReLU()
        self.l2 = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        xout = self.l2(self.re(self.bn(self.l1(x))))
        return xout


class projector_decoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(projector_decoder, self).__init__()
        self.l1 = nn.Linear(in_dim, in_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.re = nn.ReLU()
        self.l2 = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        xout = self.l2(self.re(self.bn(self.l1(x))))
        return xout

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            # nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # return (self.net(x)).view(-1)
        return self.net(x)

class relation_model(nn.Module):
    def __init__(self, indim, outdim=1):
        super(relation_model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, indim//2),
            nn.ReLU(),
            nn.Linear(indim//2, outdim),
            nn.Sigmoid()
        )
    def forward(self, x:torch.tensor):
        xout = (self.net(x)).view(-1)
        return xout


class VAE_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE_Encoder, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, latent_size)
        self.sigma = torch.nn.Linear(hidden_size, latent_size)
    def forward(self, x):# x: bs,input_size
        x = F.relu(self.linear(x)) #-> bs,hidden_size
        mu = self.mu(x) #-> bs,latent_size
        sigma = self.sigma(x)#-> bs,latent_size
        return mu,sigma

class VAE_Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(VAE_Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.linear1(x)) #->bs,hidden_size
        # x = torch.sigmoid(self.linear2(x)) #->bs,output_size
        x = self.linear2(x)
        return x

class VAE(torch.nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_size, hidden_size, latent_size)
        self.decoder = VAE_Decoder(latent_size, hidden_size, output_size)

        # self.decoder1 = VAE_Decoder(64, 128, 1426)

    def forward(self, x): #x: bs,input_size
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        re_x = self.decoder(z) # re_x: bs,output_size
        return re_x,z,mu,sigma
    

class VAE_onedecoder(torch.nn.Module):
    def __init__(self, input_size, decoder, latent_size, hidden_size):
        super(VAE_onedecoder, self).__init__()
        self.encoder = VAE_Encoder(input_size, hidden_size, latent_size)
        self.decoder = decoder
        # self.decoder = VAE_Decoder(latent_size, hidden_size, output_size)
    def forward(self, x): #x: bs,input_size
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        re_x = self.decoder(z) # re_x: bs,output_size
        return re_x,z,mu,sigma

def vaeloss(mu, sigma, re_x, x, alpha=0.1):
        mseloss = torch.nn.MSELoss()
        recon_loss = mseloss(re_x, x)
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        loss = alpha*KLD+recon_loss
        return loss

