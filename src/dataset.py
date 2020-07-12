import os
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from numpy import load


class MMIMDbDataset(Dataset):
    """Multimodal IMDb dataset (http://lisi1.unal.edu.co/mmimdb)"""

    def __init__(self, root_dir, dataset, split, transform=None):
        """
        Args:
            root_dir (string): Directory with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Load data
        self.full_data_path = os.path.join(root_dir, dataset)
        #self.embeddings_path = self.full_data_path+'/word2vec_emb_'+split+'.npy'
        #self.vgg_features_path = self.full_data_path+'/vgg_features_'+split+'.npy'
        #self.labels_path = self.full_data_path+'/labels_'+split+'.npy'
        self.embeddings = load(self.full_data_path+'/word2vec_emb_'+split+'.npy')
        self.vgg_features = load(self.full_data_path+'/vgg_features_'+split+'.npy')
        self.labels = load(self.full_data_path+'/labels_'+split+'.npy')
        
        if split == 'train':
            self.data_len = 15552
        elif split == 'dev':
            self.data_len = 2608
        else:
            self.data_len = 7799
        
    def __len__(self):
        #return self.data_len
        return self.labels.shape[0]

    def __getitem__(self, idx):
        
        #text_feature = load(self.embeddings_path)[idx]
        #visual_feature = load(self.vgg_features_path)[idx]
        #labels = load(self.labels_path)[idx]
        
        text_feature = self.embeddings[idx]
        visual_feature = self.vgg_features[idx]
        labels = self.labels[idx]

        sample = {'image': visual_feature,
                  'input_ids': text_feature,
                  "label": torch.FloatTensor(labels)}

        return sample
