import os
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset
from gensim.models import KeyedVectors

Image.MAX_IMAGE_PIXELS = 1000000000


class MMIMDbDataset(Dataset):
    """Multimodal IMDb dataset (http://lisi1.unal.edu.co/mmimdb)"""

    def __init__(self, root_dir, dataset, split, transform=None):
        """
        Args:
            jsonl_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # Metadata
        self.full_data_path = os.path.join(root_dir, dataset) + '/split.json'
        with open(self.full_data_path) as json_data:
            self.data_dict_raw = json.load(json_data)[split]
        
        plots = []
        image_names = []
        genres = []

        for id in self.data_dict_raw:
            with open(os.path.join(root_dir, dataset)+"/dataset/"+str(id)+'.json') as json_data:
                movie = json.load(json_data)
            plots.append(movie['plot'][0])
            genres.append(movie['genres'])
            image_names.append(os.path.join(root_dir, dataset)+"/dataset/"+str(id)+'.jpeg')
            
        self.data_dict = pd.DataFrame({'image': image_names, 'label': genres, 'text': plots})
            
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform
        self.genres = ['Horror', 'News', 'Animation',
                       'Musical', 'Fantasy', 'Family',
                       'Romance', 'Short', 'Comedy',
                       'Film-Noir', 'Mystery', 'Thriller',
                       'Documentary', 'Crime', 'History',
                       'Biography', 'Western', 'War',
                       'Adult', 'Adventure', 'Drama',
                       'Action', 'Music', 'Sci-Fi',
                       'Sport', 'Reality-TV', 'Talk-Show']
        self.num_classes = len(self.genres)

        # word2vec model
        self.word2vec_model = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
        
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_dict.iloc[idx,0]
        #with open(img_name, 'rb') as f:
        #    image = Image.open(f)
        #    image.convert('RGB')
        image = Image.open(img_name).convert('RGB')
        
        label = self.data_dict.iloc[idx,1]
        indeces = torch.LongTensor([self.genres.index(e) for e in label])
        label = torch.nn.functional.one_hot(indeces, num_classes = self.num_classes).sum(dim=0)

        text = self.data_dict.iloc[idx,2]
        text = self.get_text_vector(text)
        text = torch.from_numpy(text).type(torch.FloatTensor)

        if self.transform:
            image = self.transform(image)

        sample = {'image': image,
                  'input_ids': text,
                  "label": label.type(torch.FloatTensor)}

        return sample
    
    def get_text_vector(self, text):
        
        token_words = text.lower().split()
        
        sum_vectors = np.zeros(300)
        
        for w in token_words:
            try:
                w_vec = self.word2vec_model[w]
                sum_vectors += w_vec
            except KeyError:
                continue
                
        return sum_vectors