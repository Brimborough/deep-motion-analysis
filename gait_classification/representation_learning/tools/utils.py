""" This file contains different utility functions that are not connected
in anyway to the networks presented, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import pickle
import gzip
import numpy as np
import os
import re
import sys
import theano
import theano.tensor as T

from collections import Counter, defaultdict
from copy import deepcopy

data_path = '/home/jona/git/deep-motion-analysis/gait_classification/data/'

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def get_labels_to_remove(n_instances, n_to_remove):
    """
    Returns an array which indicates the number of labels to remove per class.
    This ensures that labels are first removed from a the class with the maximum
    number of labeled instances. 
    """
    if (n_to_remove > sum(n_instances)):
        raise ValueError('Number of labels to remove greater than number of labeled instances')

    to_remove = np.zeros([len(n_instances)])

    if (n_to_remove == 0):
        return to_remove

    max = np.max(n_instances)

    while (n_to_remove > 0):
        max_idx = np.where(n_instances == max)[0]
        mask = np.array(len(n_instances)*[True])
        mask[max_idx] = False

        if (n_to_remove < len(max_idx)):
            max_idx = max_idx[:n_to_remove]

        # Number of items that would be removed to set all max items equal to second largest smallest number
        mmax = np.max(n_instances[mask]) if (len(n_instances[mask]) > 0) else max-(n_to_remove / len(max_idx))
        remove_it = np.min([(n_to_remove / len(max_idx)), (max - mmax)])

        to_remove[max_idx] += remove_it
        n_instances[max_idx] -= remove_it

        n_to_remove -= len(max_idx) * remove_it
        max -= remove_it

    return to_remove

def remove_labels(rng, one_hot_labels, n_labels_to_remove):
    """
    This is used to create an (artifical) semi-supervised learning environment.
    By convention, unlabeled data is marked as a vector of zeros in lieu of a one-hot-vector
    """

    n_datapoints = one_hot_labels.shape[0]
    n_labeled_datapoints = int(np.sum(np.sum(one_hot_labels, axis=1)))

    n_classes = one_hot_labels.shape[1]
    n_instances_per_class = np.sum(one_hot_labels, axis=0)

    if (n_datapoints != n_labeled_datapoints):
        raise ValueError('Received unlabeled instances')

    label_flags = []
    for i in xrange(n_classes):
        mask = (one_hot_labels[:,i] == 1)
        # indices of datapoints belonging to class i
        label_flags.append(np.where(mask == True)[0])

    n_to_remove = get_labels_to_remove(deepcopy(n_instances_per_class), n_labels_to_remove).astype(int)

    # remove labels
    for id, lf in enumerate(label_flags):
        rng.shuffle(lf)

        # Randomnly remove labels to create a semi-supervised setting
        unlabeled_points = lf[0:n_to_remove[id]].reshape(n_to_remove[id], 1)

        # Remove labels
        one_hot_labels[unlabeled_points] = 0

    return one_hot_labels

def fair_split(rng, data, one_hot_labels, proportions):
    """
    Splits a dataset in parts given by the percentage in proportions. This split is done
    in a way that ensures the original balance between classes in every part.
    This can be important in classificaton, for instance
    """

    if (len(proportions) == 1):
        return [(data, one_hot_labels)]
    if (np.sum(proportions) != 1.0):
        raise ValueError('Proportions must sum up to one.')

    n_classes = one_hot_labels.shape[1]
    n_instances_per_class = np.sum(one_hot_labels, axis=0).astype(int)
    n_splits = len(proportions)

    n_instances_per_split = np.array([(p*n_instances_per_class).astype(int) for p in proportions])#.astype(float)
    # In case of uneven splits
    n_instances_per_split[0] += n_instances_per_class - np.sum(n_instances_per_split, axis=0)

    n_instances_per_split = np.cumsum(n_instances_per_split, axis=0)
    datasets = [[] for n in xrange(n_splits)]

    for i in xrange(n_classes):
        mask = (one_hot_labels[:,i] == 1)

        # indices of datapoints belonging to class i
        label_flags = np.where(mask == True)[0]
        rng.shuffle(label_flags)

        splits = np.split(label_flags, n_instances_per_split[:, i])
        for id, d in enumerate(datasets):
            d += splits[id].tolist()

    map(rng.shuffle, datasets)

    for id, d in enumerate(datasets):
        datasets[id] = (data[d], one_hot_labels[d])

    return datasets

def random_split(rng, data, one_hot_labels, proportions):
    """
    Splits a dataset in parts given by the percentage in proportions. This split is done
    in a way that ensures the original balance between classes in every part.
    This can be important in classificaton, for instance
    """

    if (len(proportions) == 1): 
        return [(data, one_hot_labels)] 
    if (np.sum(proportions) != 1.0):
        raise ValueError('Proportions must sum up to one.')

    n_datapoints = data.shape[0]
    proportions = np.cumsum(proportions)

    # Random split
    shuffled = (zip(data, one_hot_labels))
    rng.shuffle(shuffled)
    X, Y = map(np.array, zip(*shuffled))
    datasets = []
    split_idx = [0]

    for split in proportions:
        split_idx.append(int(split * n_datapoints))

    # In case of uneven splits
    split_idx[0] += n_datapoints - (split_idx[-1])

    for sid in xrange(1, len(split_idx)):
        datasets.append((X[split_idx[sid-1]:split_idx[sid]], 
                         Y[split_idx[sid-1]:split_idx[sid]]))

    return datasets


def load_hdm05(rng, split=(0.6, 0.2, 0.2), fair=True, filename = data_path + 'hdm05_easy/data_hdm05.npz'):

    sys.stdout.write('... loading data\n')

    data = np.load(filename)

    clips = data['clips'].swapaxes(1, 2)
    classes = data['classes']


    # Remove unlabeled data
    X = clips[classes != -1]
    Y = classes[classes != -1]

    # Set up labels
    Y = np.eye(len(np.unique(Y)))[Y]

    # Set up data
    X = X[:,:-4].astype(theano.config.floatX)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
    Xmean[:,-3:] = 0.0

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
    Xstd[:,-3:-1] = X[:,-3:-1].std()
    Xstd[:,-1:  ] = X[:,-1:  ].std()

    X = (X - Xmean) / (Xstd + 1e-10)

    # Randomise data
    I = np.arange(len(X))
    rng.shuffle(I) 

    X = X[I].astype(theano.config.floatX)
    Y = Y[I].astype(theano.config.floatX)

    # Split data and keep classes balanced
    datasets = fair_split(rng, X, Y, split)

    sys.stdout.write('... done\n')

    return datasets

def load_hdm05_actors(rng, dir = 'hdm05_easy/', n_classes = 26, train=True):

    sys.stdout.write('... loading data\n')

    match = re.findall('(.*_)(.*)/$', dir)[0]

    get_filename = lambda actor: data_path + dir + 'data_' + match[0] + \
                                 actor + '_' + match[1] + '.npz'

    to_float         = lambda x: x.astype(theano.config.floatX)
    get_xy           = lambda d: [d['clips'].swapaxes(1, 2)[:,:-4], d['classes']]
    get_mean_std     = lambda d: [d['Xmean'], d['Xstd']]
    get_one_hot      = lambda d: [d[0], one_hot[d[1]]]
    join_x           = lambda x1, x2: np.concatenate([x1, x2], axis=0)
    join             = lambda d1, d2: [join_x(d1[0], d2[0]), join_x(d1[1], d2[1])]
    normalise        = lambda d: [(d[0] - mean) / (std + 1e-10), d[1]]
    remove_unlabeled = lambda d: [d[0][d[1] != -1 ], d[1][d[1] != -1 ]]

    # Calls functions above
    process = lambda d: get_one_hot(normalise(remove_unlabeled(get_xy(d))))

    mean, std = get_mean_std(np.load(data_path + dir + 'moments.npz'))
    one_hot   = np.eye(n_classes)

    data_bd = process(np.load(get_filename('bd')))
    data_bk = process(np.load(get_filename('bk')))
    data_dg = process(np.load(get_filename('dg')))
    data_mm = process(np.load(get_filename('mm')))
    data_tr = process(np.load(get_filename('tr')))

    if train:
#        This leads to an unfair split where we're testing on data
#        that is not in the training set
#        train_set = join(data_bd, data_mm)
        train_set = join(data_tr, data_mm)
        valid_set = data_bk
    else:
        train_set = join(join(data_tr, data_mm), data_bk)
        valid_set = []

#    test_set = join(data_tr, data_dg)
    test_set = join(data_dg, data_bd)

    # Randomise training data
    I = np.arange(len(train_set[0]))
    rng.shuffle(I) 

    train_set[0] = train_set[0][I]
    train_set[1] = train_set[1][I]

    sys.stdout.write('... done\n')

    return [train_set, valid_set, test_set]

def load_hdm05_easy(rng, split = (0.6, 0.2, 0.2), fair = True):
    return load_hdm05(rng = rng, split = split, fair = fair, 
                      filename = data_path + 'hdm05_easy/data_hdm05_easy.npz')

def load_hdm05_original(rng, split = (0.6, 0.2, 0.2), fair = True):
    return load_hdm05(rng = rng, split = split, fair = fair, 
                      filename = data_path + 'hdm05_original/data_hdm05_original.npz')

def load_hdm05_65(rng, split = (0.6, 0.2, 0.2), fair = True):
    return load_hdm05(rng = rng, split = split, fair = fair, 
                      filename = data_path + 'hdm05_65/data_hdm05_65.npz')

def load_hdm05_actors_65(rng, train=True):
    return load_hdm05_actors(rng, dir = 'hdm05_65/', n_classes = 65, train=train)

def load_hdm05_actors_easy(rng, train=True):
    return load_hdm05_actors(rng, dir = 'hdm05_easy/', n_classes = 26, train=train)

def load_hdm05_actors_original(rng, train=True):
    return load_hdm05_actors(rng, dir = 'hdm05_original/', n_classes = 139, train=train)

def load_styletransfer(rng, split=(0.6, 0.2, 0.2), labels='combined'):

    sys.stdout.write('... loading data\n')

    data = np.load(data_path + 'styletransfer/data_styletransfer.npz')

    clips = data['clips'].swapaxes(1, 2)
    X = clips[:,:-4]

    #(Motion, Styles)
    classes = data['classes']

    # get mean and std
    preprocessed = np.load(data_path + '/styletransfer/styletransfer_preprocessed.npz')

    Xmean = preprocessed['Xmean']
    Xmean = Xmean.reshape(1,len(Xmean),1)
    Xstd  = preprocessed['Xstd']
    Xstd = Xstd.reshape(1,len(Xstd),1)

    X = (X - Xmean) / (Xstd + 1e-10)

    # Motion labels in one-hot vector format
    Y = np.load(data_path + 'styletransfer/styletransfer_one_hot.npz')[labels]

    # Randomise data
    I = np.arange(len(X))
    rng.shuffle(I)

    X = X[I].astype(theano.config.floatX)
    Y = Y[I].astype(theano.config.floatX)

    datasets = fair_split(rng, X, Y, split)

    sys.stdout.write('... done\n')

    return datasets

def load_cmu(rng, filename = data_path + 'cmu/data_cmu.npz'):

    sys.stdout.write('... loading data\n')

    data = np.load(filename)

    clips = data['clips']

    clips = np.swapaxes(clips, 1, 2)
    X = clips[:,:-4].astype(theano.config.floatX)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
    Xmean[:,-3:] = 0.0

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
    Xstd[:,-3:-1] = X[:,-3:-1].std()
    Xstd[:,-1:  ] = X[:,-1:  ].std()

    X = (X - Xmean) / (Xstd + 1e-10)

    # Randomise data
    I = np.arange(len(X))
    rng.shuffle(I) 
    X = X[I]

    sys.stdout.write('... done\n')

    return [(X,)]

def load_cmu_small(rng):
    return load_cmu(rng=rng, filename = data_path + 'cmu/data_cmu_small.npz')

def load_mnist(rng):
    ''' Loads the MNIST dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    dataset = data_path + 'mnist/mnist.pkl.gz'

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        sys.stdout.write(('Downloading data from %s\n') % origin)
        urllib.request.urlretrieve(origin, dataset)

    sys.stdout.write('... loading data\n')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a np.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # np.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def one_hot_labels(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy

        n_datapoints = data_y.shape[0]
        # Convert to one_hot_labels
        # Digits 0-9: 10 classes
        one_hot_labels = np.zeros([n_datapoints, 10])
        one_hot_labels[np.arange(n_datapoints), data_y] = 1

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return (np.asarray(data_x, dtype=theano.config.floatX), np.asarray(one_hot_labels, dtype=theano.config.floatX)) #T.cast(shared_y, 'int32')

    train_set = one_hot_labels(train_set)
    valid_set = one_hot_labels(valid_set)
    test_set  = one_hot_labels(test_set)

    datasets = [train_set, valid_set, test_set]
    return datasets
