import h5py
import json
import logging
import math
import os
import numpy
import re
import sys
from collections import OrderedDict, Counter
from fuel.datasets import H5PYDataset
#from fuel.utils import find_in_data_path
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from matplotlib import pyplot
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def normalizeText(text):
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text.split()

def resize_and_crop_image(input_file, output_box=[224, 224], fit=True):
        # https://github.com/BVLC/caffe/blob/master/tools/extra/resize_and_crop_images.py
        '''Downsample the image.
        '''
        img = Image.open(input_file)
        #img.save("orig_"+input_file.split('/')[-1])
        box = output_box
        # preresize image with factor 2, 4, 8 and fast algorithm
        factor = 1
        while img.size[0] / factor > 2 * box[0] and img.size[1] * 2 / factor > 2 * box[1]:
            factor *= 2
        if factor > 1:
            img.thumbnail(
                (img.size[0] / factor, img.size[1] / factor), Image.NEAREST)

        # calculate the cropping box and get the cropped part
        if fit:
            x1 = y1 = 0
            x2, y2 = img.size
            wRatio = 1.0 * x2 / box[0]
            hRatio = 1.0 * y2 / box[1]
            if hRatio > wRatio:
                y1 = int(y2 / 2 - box[1] * wRatio / 2)
                y2 = int(y2 / 2 + box[1] * wRatio / 2)
            else:
                x1 = int(x2 / 2 - box[0] * hRatio / 2)
                x2 = int(x2 / 2 + box[0] * hRatio / 2)
            img = img.crop((x1, y1, x2, y2))

        # Resize the image with best quality algorithm ANTI-ALIAS
        img = img.resize(box, Image.ANTIALIAS).convert('RGB')
        #img = numpy.asarray(img, dtype='float32')
        return img

def get_image_feature(feature_extractor, image):
    with torch.no_grad():
        feature_images = feature_extractor.features(image)
        feature_images = feature_extractor.avgpool(feature_images)
        feature_images = torch.flatten(feature_images, 1)
        feature_images = feature_extractor.classifier[0](feature_images)
    
    return feature_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conf_file = sys.argv[1] if len(sys.argv) > 1 else None
with open(conf_file) as f:
    locals().update(json.load(f))

with open('list.txt', 'r') as f:
    files = f.read().splitlines()

## Load data and define vocab ##
logger.info('Reading json and jpeg files...')
movies = []
vocab_counts = []
feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
for param in feature_extractor.features.parameters():
    param.requires_grad = False
    
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

for i, file in enumerate(files):

    with open(file) as f:
        data = json.load(f)
        data['imdb_id'] = file.split('/')[-1].split('.')[0]
        # if 'plot' in data and 'plot outline' in data:
        #    data['plot'].append(data['plot outline'])
        im_file = file.replace('json', 'jpeg')
        if all([k in data for k in ('genres', 'plot')] + [os.path.isfile(im_file)]):
            plot_id = numpy.array([len(p) for p in data['plot']]).argmax()
            data['plot'] = normalizeText(data['plot'][plot_id])
            
            if len(data['plot']) > 0:
                vocab_counts.extend(data['plot'])
                #data['cover'] = resize_and_crop_image(im_file, img_size)
                #Image.fromarray(data['cover'].save(im_file.split('/')[-1])
                img = resize_and_crop_image(im_file, img_size)
                img = preprocess(img)
                img = img.unsqueeze(0)
                feature = get_image_feature(feature_extractor, img)
                data['vgg_features'] = feature.squeeze(0).numpy()
                movies.append(data)
    logger.info('{0:05d} out of {1:05d}: {2:02.2f}%'.format(
        i, len(files), float(i) / len(files) * 100))

logger.info('done reading files.')

vocab_counts = OrderedDict(Counter(vocab_counts).most_common())
vocab = ['_UNK_'] + [v for v in vocab_counts.keys()]
googleword2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
#ix_to_word = dict(zip(range(len(vocab)), vocab))
#word_to_ix = dict(zip(vocab, range(len(vocab))))
#lookup = numpy.array([googleword2vec[v] for v in vocab if v in googleword2vec])
#numpy.save('metadata.npy', {'ix_to_word': ix_to_word,
#                            'word_to_ix': word_to_ix,
#                            'vocab_size': len(vocab),
#                            'lookup': lookup})


# Define train, dev and test subsets
counts = OrderedDict(
    Counter([g for m in movies for g in m['genres']]).most_common())
target_names = list(counts.keys())[:n_classes]

le = MultiLabelBinarizer()
Y = le.fit_transform([m['genres'] for m in movies])
labels = numpy.nonzero(le.transform([[t] for t in target_names]))[1]

B = numpy.copy(Y)
rng = numpy.random.RandomState(rng_seed)
train_idx, dev_idx, test_idx = [], [], []
for l in labels[::-1]:
    t = B[:, l].nonzero()[0]
    t = rng.permutation(t)
    n_test = int(math.ceil(len(t) * test_size))
    n_dev = int(math.ceil(len(t) * dev_size))
    n_train = len(t) - n_test - n_dev
    test_idx.extend(t[:n_test])
    dev_idx.extend(t[n_test:n_test + n_dev])
    train_idx.extend(t[n_test + n_dev:])
    B[t, :] = 0

indices = numpy.concatenate([train_idx, dev_idx, test_idx])
nsamples = len(indices)
nsamples_train, nsamples_dev, nsamples_test = len(
    train_idx), len(dev_idx), len(test_idx)

# Obtain feature vectors and text sequences
sequences = []
X = numpy.zeros((indices.shape[0], textual_dim), dtype='float32')
for i, idx in enumerate(indices):
    words = movies[idx]['plot']
    #sequences.append([word_to_ix[w] if w in vocab else unk_idx for w in words])
    X[i] = numpy.array([googleword2vec[w]
                        for w in words if w in googleword2vec]).mean(axis=0)

del googleword2vec

# get n-grams representation
'''
sentences = [' '.join(m['plot']) for m in movies]
ngram_vectorizer = TfidfVectorizer(
    analyzer='char', ngram_range=(3, 3), min_df=2)
ngrams_feats = ngram_vectorizer.fit_transform(sentences).astype('float32')
word_vectorizer = TfidfVectorizer(min_df=10)
wordgrams_feats = word_vectorizer.fit_transform(sentences).astype('float32')
'''

# Store data in the hdf5 file
f_train = h5py.File('multimodal_imdb_train.hdf5', mode='w')
f_dev = h5py.File('multimodal_imdb_dev.hdf5', mode='w')
f_test = h5py.File('multimodal_imdb_test.hdf5', mode='w')

dtype = h5py.special_dtype(vlen=numpy.dtype('int32'))

features_train = f_train.create_dataset('features', (nsamples_train, textual_dim), dtype='float32')
vgg_features_train = f_train.create_dataset(
    'vgg_features', (nsamples_train, 4096), dtype='float32')
genres_train = f_train.create_dataset('labels', (nsamples_train, n_classes), dtype='int32')

features_dev = f_dev.create_dataset('features', (nsamples_dev, textual_dim), dtype='float32')
vgg_features_dev = f_dev.create_dataset(
    'vgg_features', (nsamples_dev, 4096), dtype='float32')
genres_dev = f_dev.create_dataset('labels', (nsamples_dev, n_classes), dtype='int32')

features_test = f_test.create_dataset('features', (nsamples_test, textual_dim), dtype='float32')
vgg_features_test = f_test.create_dataset(
    'vgg_features', (nsamples_test, 4096), dtype='float32')
genres_test = f_test.create_dataset('labels', (nsamples_test, n_classes), dtype='int32')


#three_grams = f.create_dataset(
#    'three_grams', (nsamples, ngrams_feats.shape[1]), dtype='float32')
#word_grams = f.create_dataset(
#    'word_grams', (nsamples, wordgrams_feats.shape[1]), dtype='float32')
#images = f.create_dataset(
#    'images', [nsamples, num_channels] + img_size[::-1], dtype='int32')
#seqs = f.create_dataset('sequences', (nsamples,), dtype=dtype)
#imdb_ids = f.create_dataset('imdb_ids', (nsamples,), dtype="S7")
#imdb_ids[...] = numpy.asarray([m['imdb_id']
#                               for m in movies], dtype='S7')[indices]

features_train[...] = X[:nsamples_train, :]
features_dev[...] = X[nsamples_train:nsamples_train+nsamples_dev, :]
features_test[...] = X[nsamples_train+nsamples_dev:, :]

for i, idx in enumerate(train_idx):
    vgg_features_train[i] = movies[idx]['vgg_features']

for i, idx in enumerate(dev_idx):
    vgg_features_dev[i] = movies[idx]['vgg_features']

for i, idx in enumerate(test_idx):
    vgg_features_test[i] = movies[idx]['vgg_features']
    
genres_train[...] = Y[train_idx][:, labels]
genres_dev[...] = Y[dev_idx][:, labels]
genres_test[...] = Y[test_idx][:, labels]

'''
#for i, idx in enumerate(indices):
#    vgg_features[i] = movies[idx]['vgg_features']
#seqs[...] = sequences
genres[...] = Y[indices][:, labels]
#three_grams[...] = ngrams_feats[indices].todense()
#word_grams[...] = wordgrams_feats[indices].todense()
genres.attrs['target_names'] = json.dumps(target_names)
features.dims[0].label = 'batch'
features.dims[1].label = 'features'
#three_grams.dims[0].label = 'batch'
#three_grams.dims[1].label = 'features'
#word_grams.dims[0].label = 'batch'
#word_grams.dims[1].label = 'features'
#imdb_ids.dims[0].label = 'batch'
genres.dims[0].label = 'batch'
genres.dims[1].label = 'classes'
vgg_features.dims[0].label = 'batch'
vgg_features.dims[1].label = 'features'
#images.dims[0].label = 'batch'
#images.dims[1].label = 'channel'
#images.dims[2].label = 'height'
#images.dims[3].label = 'width'
'''

'''
split_dict = {
    'train': {
        'features': (0, nsamples_train),
        #'three_grams': (0, nsamples_train),
        #'sequences': (0, nsamples_train),
        #'images': (0, nsamples_train),
        'vgg_features': (0, nsamples_train),
        #'imdb_ids': (0, nsamples_train),
        #'word_grams': (0, nsamples_train),
        'genres': (0, nsamples_train)},
    'dev': {
        'features': (nsamples_train, nsamples_train + nsamples_dev),
        #'three_grams': (nsamples_train, nsamples_train + nsamples_dev),
        #'sequences': (nsamples_train, nsamples_train + nsamples_dev),
        #'images': (nsamples_train, nsamples_train + nsamples_dev),
        'vgg_features': (nsamples_train, nsamples_train + nsamples_dev),
        #'imdb_ids': (nsamples_train, nsamples_train + nsamples_dev),
        #'word_grams': (nsamples_train, nsamples_train + nsamples_dev),
        'genres': (nsamples_train, nsamples_train + nsamples_dev)},
    'test': {
        'features': (nsamples_train + nsamples_dev, nsamples),
        #'three_grams': (nsamples_train + nsamples_dev, nsamples),
        #'sequences': (nsamples_train + nsamples_dev, nsamples),
        #'images': (nsamples_train + nsamples_dev, nsamples),
        'vgg_features': (nsamples_train + nsamples_dev, nsamples),
        #'imdb_ids': (nsamples_train + nsamples_dev, nsamples),
        #'word_grams': (nsamples_train + nsamples_dev, nsamples),
        'genres': (nsamples_train + nsamples_dev, nsamples)}
}
'''

#f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
f_train.flush()
f_train.close()

f_dev.flush()
f_dev.close()

f_test.flush()
f_test.close()
