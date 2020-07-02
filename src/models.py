import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel


class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        # Weights hidden state modality 1
        weights_hidden1 = torch.Tensor(size_out, size_in1)
        self.weights_hidden1 = nn.Parameter(weights_hidden1)

        # Weights hidden state modality 2
        weights_hidden2 = torch.Tensor(size_out, size_in2)
        self.weights_hidden2 = nn.Parameter(weights_hidden2)

        # Weight for sigmoid
        weight_sigmoid = torch.Tensor(size_out*2)
        self.weight_sigmoid = nn.Parameter(weight_sigmoid)

        # initialize weights
        nn.init.kaiming_uniform_(self.weights_hidden1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weights_hidden2, a=math.sqrt(5))

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(torch.mm(x1, self.weights_hidden1.t()))
        h2 = self.tanh_f(torch.mm(x2, self.weights_hidden2.t()))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(torch.matmul(x, self.weight_sigmoid.t()))

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


class ConcatenateModel(nn.Module):

    def __init__(self, hyp_params):

        super(ConcatenateModel, self).__init__()
        self.bn1 = nn.BatchNorm1d(hyp_params.text_embedding_size+hyp_params.image_feature_size)
        self.linear1 = MaxOut(hyp_params.text_embedding_size+hyp_params.image_feature_size, hyp_params.hidden_size)
        self.bn2 = nn.BatchNorm1d(hyp_params.hidden_size)
        self.linear2 = MaxOut(hyp_params.hidden_size, hyp_params.hidden_size)
        self.bn3 = nn.BatchNorm1d(hyp_params.hidden_size)
        self.linear3 = nn.Linear(hyp_params.hidden_size, hyp_params.output_dim)
        #self.drop1 = nn.Dropout(p=hyp_params.mlp_dropout)

    def forward(self, input_ids, feature_images):
        
        x = torch.cat((input_ids, feature_images), dim=1)
        x = self.bn1(x)
        x = self.linear1(x)
        x = self.bn2(x)
        x = self.linear2(x)
        x = self.bn3(x)
        x = self.linear3(x)

        return nn.Sigmoid(x)
    