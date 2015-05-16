import theano
import theano.tensor as T
from logistic_sgd import load_data
from theanet.theanet.layer.inlayers import ElasticLayer

from PIL import Image
import scipy.misc

import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

import pdb

SIGMA = 6
ALPHA = 36
dataset='mnist.pkl.gz'
normalized_width = 20
batch_size = 25

def distortions(datasets, sigma, alpha):
    train_set_x, train_set_y = datasets[0]
    x = T.matrix('x')

    distortion_layer = ElasticLayer(
        x.reshape((batch_size, 29, 29)),
        29,
        magnitude=alpha,
        sigma=sigma
    )
    deform_fn = theano.function([x], distortion_layer.output)

    data = train_set_x.get_value(borrow=True)[:batch_size, :]

    dist_data = deform_fn(data)

    return [data.reshape((batch_size,29,29)), dist_data]

def plot_matrix_of_images(mtrx, name):
    nrow = len(mtrx)
    ncol = mtrx[0].shape[0]
    fig, axarr = plt.subplots(nrow, ncol, figsize=(ncol, nrow), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.001, wspace=0.001)
    kwargs = dict(cmap=cm.Greys_r)
    for ri in xrange(nrow):
        imgs = mtrx[ri]
        for ci in xrange(ncol):
            axarr[ri, ci].imshow(imgs[ci], **kwargs)

    plt.savefig('plots/'+name+".png",bbox_inches='tight')

if __name__ == '__main__':
    datasets = load_data(dataset, normalized_width, 29)

    first = distortions(datasets, 9, 36)
    distorted = [
        first[0],
        first[1],
        distortions(datasets, 8, 36)[1],
        distortions(datasets, 7, 36)[1],
        distortions(datasets, 6, 36)[1],
        distortions(datasets, 5, 36)[1]
    ]
    plot_matrix_of_images(distorted, 'distortions_9_to_5')

    first = distortions(datasets, 8, 36)
    distorted = [
        first[0],
        first[1],
        distortions(datasets, 8, 36)[1],
        distortions(datasets, 8, 36)[1],
        distortions(datasets, 8, 36)[1],
        distortions(datasets, 8, 36)[1]
    ]
    plot_matrix_of_images(distorted, 'distortions_8_sampled')


    # pdb.set_trace()
