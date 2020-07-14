import math
import torch
from torch import nn
import torch.nn.functional as F
#from .module import Module


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out*2, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden1(x2))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))

        return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2
    
    
class MaxOut(nn.Module):
    def __init__(self, input_dim, output_dim, num_units=2):
        super(MaxOut, self).__init__()
        self.fc1_list = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(num_units)])

    def forward(self, x): 

        return self.maxout(x, self.fc1_list)

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


class MLPGenreClassifierModel(nn.Module):

    def __init__(self, hyp_params):

        super(MLPGenreClassifierModel, self).__init__()
        if hyp_params.text_embedding_size == hyp_params.image_feature_size:
            self.bn1 = nn.BatchNorm1d(hyp_params.hidden_size)
            self.linear1 = MaxOut(hyp_params.hidden_size, hyp_params.hidden_size)
        else:
            self.bn1 = nn.BatchNorm1d(hyp_params.text_embedding_size+hyp_params.image_feature_size)
            self.linear1 = MaxOut(hyp_params.text_embedding_size+hyp_params.image_feature_size, hyp_params.hidden_size)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)
        
        self.bn2 = nn.BatchNorm1d(hyp_params.hidden_size)
        self.linear2 = MaxOut(hyp_params.hidden_size, hyp_params.hidden_size)
        self.drop2 = nn.Dropout(p=hyp_params.mlp_dropout)
        
        self.bn3 = nn.BatchNorm1d(hyp_params.hidden_size)
        self.linear3 = nn.Linear(hyp_params.hidden_size, hyp_params.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, feature_images=None):
        if feature_images is None:
            x = input_ids
        else:
            x = torch.cat((input_ids, feature_images), dim=1)
        x = self.bn1(x)
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.bn2(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = self.bn3(x)
        x = self.linear3(x)

        return self.sigmoid(x)
    

class ConcatenateModel(nn.Module):

    def __init__(self, hyp_params):

        super(ConcatenateModel, self).__init__()
        self.linear1 = MaxOut(hyp_params.text_embedding_size+hyp_params.image_feature_size, hyp_params.hidden_size)
        self.bn1 = nn.BatchNorm1d(hyp_params.hidden_size)
        self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)

        self.linear2 = MaxOut(hyp_params.hidden_size, hyp_params.hidden_size)
        self.bn2 = nn.BatchNorm1d(hyp_params.hidden_size)
        self.drop2 = nn.Dropout(p=hyp_params.mlp_dropout)
        
        self.linear3 = nn.Linear(hyp_params.hidden_size, hyp_params.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, feature_images):
        
        x = torch.cat((input_ids, feature_images), dim=1)
        x = self.drop1(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.drop2(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.linear3(x)

        return self.sigmoid(x)
    
    
class GMUModel(nn.Module):

    def __init__(self, hyp_params):

        super(GMUModel, self).__init__()
        self.hyp_params = {}
        self.hyp_params['text_embedding_size'] = hyp_params.hidden_size
        self.hyp_params['image_feature_size'] = hyp_params.hidden_size
        self.hyp_params['hidden_size'] = hyp_params.hidden_size
        self.hyp_params['output_dim'] = hyp_params.output_dim
        self.hyp_params['mlp_dropout'] = hyp_params.mlp_dropout
        self.hyp_params = dotdict(self.hyp_params)
        
        self.visual_mlp = torch.nn.Sequential(
            nn.BatchNorm1d(hyp_params.image_feature_size),
            nn.Linear(hyp_params.image_feature_size, hyp_params.hidden_size)
        )
        self.textual_mlp = torch.nn.Sequential(
            nn.BatchNorm1d(hyp_params.text_embedding_size),
            nn.Linear(hyp_params.text_embedding_size, hyp_params.hidden_size)
        )
        
        self.gmu = GatedMultimodalLayer(hyp_params.hidden_size, hyp_params.hidden_size, hyp_params.hidden_size)
        
        self.logistic_mlp = MLPGenreClassifierModel(self.hyp_params)

    def forward(self, input_ids, feature_images):
        
        x_v = self.visual_mlp(feature_images)
        x_t = self.textual_mlp(input_ids)
        x = self.gmu(x_v, x_t)
        
        return self.logistic_mlp(x)
