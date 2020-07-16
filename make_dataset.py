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
from numpy import save
from numpy import asarray


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
    X[i] = numpy.array([googleword2vec[w]
                        for w in words if w in googleword2vec]).mean(axis=0)

del googleword2vec


# Save embeddings

mat = asarray(X[:nsamples_train, :])
save('../mmimdb/data_word2vec/w2v_train.npy', mat)

mat = asarray(X[nsamples_train:nsamples_train+nsamples_dev, :])
save('../mmimdb/data_word2vec/w2v_dev.npy', mat)

mat = asarray(X[nsamples_train+nsamples_dev:, :])
save('../mmimdb/data_word2vec/w2v_test.npy', mat)

# Save image features

vgg_features = np.zeros((len(train_idx), 4096))
for i, idx in enumerate(train_idx):
    vgg_features[i] = movies[idx]['vgg_features']
mat = asarray(vgg_features)
save('../mmimdb/data_word2vec/vgg_train.npy', mat)

vgg_features = np.zeros((len(dev_idx), 4096))
for i, idx in enumerate(dev_idx):
    vgg_features[i] = movies[idx]['vgg_features']
mat = asarray(vgg_features)
save('../mmimdb/data_word2vec/vgg_dev.npy', mat)

vgg_features = np.zeros((len(test_idx), 4096))
for i, idx in enumerate(test_idx):
    vgg_features[i] = movies[idx]['vgg_features']
mat = asarray(vgg_features)
save('../mmimdb/data_word2vec/vgg_test.npy', mat)

# Save labels

mat = asarray(Y[train_idx][:, labels])
save('../mmimdb/data_word2vec/labels__train.npy', mat)

mat = asarray(Y[dev_idx][:, labels])
save('../mmimdb/data_word2vec/labels__dev.npy', mat)

mat = asarray(Y[test_idx][:, labels])
save('../mmimdb/data_word2vec/labels__test.npy', mat)


# Plot distribution
cm = numpy.zeros((n_classes, n_classes), dtype='int')
for i, l in enumerate(labels):
    cm[i] = Y[Y[:, l].nonzero()[0]].sum(axis=0)[labels]

cmap = pyplot.cm.Blues
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
for i in range(len(target_names)):
    cm_normalized[i, i] = 0
pyplot.imshow(cm_normalized, interpolation='nearest', cmap=cmap, aspect='auto')
for i, cas in enumerate(cm):
    for j, c in enumerate(cas):
        if c > 0:
            pyplot.text(j - .2, i + .2, c, fontsize=4)
pyplot.title('Shared labels', fontsize='smaller')
pyplot.colorbar()
tick_marks = numpy.arange(len(target_names))
pyplot.xticks(tick_marks, target_names, rotation=90)
pyplot.yticks(tick_marks, target_names)
pyplot.tight_layout()
pyplot.savefig('distribution.pdf')
pyplot.close()