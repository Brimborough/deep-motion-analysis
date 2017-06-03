""" This file contains different utility functions that are not connected
in anyway to the networks presented, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import pickle
import gzip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys
import theano
import theano.tensor as T

from collections import Counter, defaultdict
from copy import deepcopy

data_path = '/home/USER_NAME/deep-motion-analysis/gait_classification/data/'

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


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


def load_hdm05(rng, split=(0.6, 0.2, 0.2), fair=True, filename = data_path + 'hdm05/data_hdm05.npz'):

    sys.stdout.write('... loading data\n')

    data = np.load(filename)

    clips = data['clips'].swapaxes(1, 2)
    classes = data['classes']


    # Remove unlabeled data
    X = clips[classes != -1]
    Y = classes[classes != -1]

    print classes.shape

    print classes[:10]

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

    return datasets

def load_hdm05_small(rng, split = (0.6, 0.2, 0.2), fair = True):
    return load_hdm05(rng = rng, split = split, fair = fair, 
                      filename = data_path + 'hdm05/data_hdm05_small.npz')

def load_hdm05_easy(rng, split = (0.6, 0.2, 0.2), fair = True):
    return load_hdm05(rng = rng, split = split, fair = fair, 
                      filename = data_path + 'hdm05/data_hdm05_easy.npz')

def load_hdm05_easy_small(rng, split = (0.6, 0.2, 0.2), fair = True):
    return load_hdm05(rng = rng, split = split, fair = fair, 
                      filename = data_path + 'hdm05/data_hdm05_easy_small.npz')

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
    
    return datasets

def load_cmu(rng, filename='../data/cmu/data_cmu.npz'):

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

    #Xstd[np.where(Xstd == 0)] = 1

    X = (X - Xmean) / (Xstd + 1e-10)

    # Randomise data
    #I = np.arange(len(X))
    #rng.shuffle(I); 
    #X = X[I]
    #Xstd = 1.
    #Xmean = 0.

    return [(X,)], Xstd, Xmean

def load_terrain(rng, filename='../data/data_edin_terrain.npz'):

    sys.stdout.write('... loading data\n')

    data = np.load(filename)

    clips = data['clips']

    clips = np.swapaxes(clips, 1, 2)
    X = clips[:,:-4].astype(theano.config.floatX)
    X = X[:,3:].astype(theano.config.floatX)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
    Xmean[:,-3:] = 0.0

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
    Xstd[:,-3:-1] = X[:,-3:-1].std()
    Xstd[:,-1:  ] = X[:,-1:  ].std()

    X = (X - Xmean) / (Xstd + 1e-10)

    return [(X,)], Xstd, Xmean

def load_terrain_w_footsteps(rng, filename='../data/data_edin_terrain.npz'):

    sys.stdout.write('... loading data\n')

    data = np.load(filename)

    clips = data['clips']

    clips = np.swapaxes(clips, 1, 2)
    X = clips[:,:-4].astype(theano.config.floatX)
    X = X[:,3:].astype(theano.config.floatX)

    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
    Xmean[:,-3:] = 0.0

    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
    Xstd[:,-3:-1] = X[:,-3:-1].std()
    Xstd[:,-1:  ] = X[:,-1:  ].std()

    X = (X - Xmean) / (Xstd + 1e-10)
    C = clips[:,-4:] - 0.5
    X = np.concatenate([X, C.astype(theano.config.floatX)], axis=1)

    return [(X,)], Xstd, Xmean


def load_hdm05_generation(rng, filename = data_path + 'hdm05_original/data_hdm05_original.npz'):

    sys.stdout.write('... loading data\n')

    data = np.load(filename)

    clips = data['clips'].swapaxes(1, 2)

    # Remove unlabeled data
    X = clips[classes != -1]

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

    return [(X,)], Xstd, Xmean

def load_locomotion(rng, filename='../data/cmu/data_edin_locomotion_processed.npz'):

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

    return [(X,)], Xstd, Xmean

def load_locomotion_w_footsteps(rng, filename='../data/cmu/data_edin_locomotion_processed.npz'):

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
    C = clips[:,-4:] - 0.5
    X = np.concatenate([X, C.astype(theano.config.floatX)], axis=1)

    return [(X,)], Xstd, Xmean

def load_cmu_small(rng):
    return load_cmu(rng=rng, filename='../data/cmu/data_cmu_small.npz')

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

def plot_motion():
    data = np.load(data_path + 'styletransfer/data_styletransfer.npz')
    X = data['clips'].swapaxes(1, 2)[0][:-4,:][:,0]
    xs = X[::3]
    ys = X[1::3]
    zs = X[2::3]

    scale = 1.5*((len(animations))/2)
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-scale*30, scale*30)
    ax.set_zlim3d( 0, scale*60)
    ax.set_ylim3d(-scale*30, scale*30)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])
