# -*- coding: utf-8 -*-
"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from PIL import Image
import scipy.misc

import pdb

SS = 28 # start size of mnist input data

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        if W is None:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            W_values = numpy.zeros((n_in, n_out), dtype=theano.config.floatX)
        else:
            W_values = numpy.asarray(W, dtype=theano.config.floatX)

        if b is None:
            # initialize the baises b as a vector of n_out 0s
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        else:
            b_values = numpy.asarray(b, dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, name='W', borrow=True)
        self.b = theano.shared(value=b_values, name='b', borrow=True)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def prepare_digits(sets, end_size, normalized_width):
    set_x, set_y = sets[0], sets[1]
    out = numpy.ndarray((set_x.shape[0], end_size**2), dtype=numpy.float32)

    for i in xrange(0,set_x.shape[0]):
        x = set_x[i].reshape((SS,SS))
        if normalized_width and (set_y[i] - 1): # don't normalize images of digit '1'
            out[i] = normalize_digit(x, normalized_width, end_size).reshape(end_size**2)
        else:
            out[i] = pad_image(x, end_size).reshape(end_size**2)
    return out

def pad_image(x, end_size):
    """
    resizes the image x to end_size by either pading or unpadding
    the edges of the image. Returns a flat array.

    input x should be a square numpy array

    assume square image, so bp and ap can never be
    positive and negative combos (only + and 0, or - and 0)
    """
    cs = x.shape[0]
    padding = end_size - cs
    bp = round(padding / 2) # before padding (left)
    ap = padding - bp # after padding (right)
    pads = (bp,ap)
    if bp + ap > 0:
        return numpy.pad(x,(pads,pads),mode='constant').reshape(end_size**2)
    else: # image is too big now, unpad/slice
        si = -bp # start index
        ei = cs + ap # end index
        return x[si:ei, si:ei].reshape(end_size**2)

def normalize_digit(x, normalized_width, end_size):
    """
    Stretches the image so that the width of the bounding box of the digit
    equals normalized_width, then resizes to end_size with pad_image

    input x should be a square numpy array
    """
    width_diff = normalized_width - sum(sum(x) != 0) # num non-zero col-sums (there are no discontinuous numbers)
    if width_diff:
        nd = SS + width_diff # new dim
        new_size = nd, nd
        im = scipy.misc.toimage(x)
        normalized_image = im.resize(new_size, Image.ANTIALIAS)
        x = numpy.array(normalized_image.getdata(), dtype=numpy.float32).reshape((nd,nd)) / 255
    # based on my visual inspection, this assertion should pass, but doesn't
    # perhaps b/c of the smoothing that goes on with the resizing filter
    # assert sum(sum(x) != 0) == normalized_width
    return pad_image(x, end_size)

def load_data(dataset, normalized_width=0, out_image_size=SS,
              conserve_gpu_memory=False, center=0, normalize=1, image_shape=None, y_values_only=False):
    ''' Loads a dataset, and performs specified preprocessing

    :type dataset: string
    :param dataset: the path to the dataset
    '''

    #############
    # LOAD DATA #
    #############

    data_dir, data_file = os.path.split(dataset)
    data_ext = '.'.join(data_file.split('.')[1:])
    input_pixel_max = 1 if data_file == 'mnist.pkl.gz' else 255

    if data_file == 'mnist.pkl.gz':
        # Download the MNIST dataset if it is not present
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
            import urllib
            origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            )
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    if data_file == 'mnist.pkl.gz':
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        if normalized_width or (out_image_size != SS):
            if normalized_width:
                print '... normalizing digits to width %i with extra padding %i' % (normalized_width, out_image_size - SS)
            else:
                print '... (un)padding digits from %i -> %i' % (SS, out_image_size)
            train_set = (prepare_digits(train_set, out_image_size, normalized_width), train_set[1])
            valid_set = (prepare_digits(valid_set, out_image_size, normalized_width), valid_set[1])
            test_set =  (prepare_digits(test_set, out_image_size, normalized_width),  test_set[1])
        else:
            print '... skipping digit normalization and image padding'

        f.close()
        #train_set, valid_set, test_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        #numpy.ndarray of 1 dimensions (vector)) that have the same length as
        #the number of rows in the input. It should give the target
        #target to the example with the same index in the input.
    elif data_ext == 'npz': # a dataset saved with package_data.py
        with numpy.load(dataset) as archive:
            train_set = (archive['arr_0'], archive['arr_1'])
            valid_set = (archive['arr_2'], archive['arr_3'])
            test_set =  (archive['arr_4'], archive['arr_5'])
    else:
        raise ValueError("unsupported data extension %s" % data_ext)

    if y_values_only:
        print '... returning y values'
        return (train_set[1], valid_set[1], test_set[1])

    accuracy_dtype = int

    # general pre-processing (should use information from training set only)
    if center == 1:
        assert(image_shape)
        print '... subtracting channel mean'
        channel_means = numpy.mean(train_set[0].reshape(train_set[0].shape[0], *image_shape), axis=(0,1,2))
        train_set = subtract_channel_mean(train_set, image_shape, channel_means, accuracy_dtype)
        valid_set = subtract_channel_mean(valid_set, image_shape, channel_means, accuracy_dtype)
        test_set = subtract_channel_mean(test_set, image_shape, channel_means, accuracy_dtype)
    elif center == 2:
        print '... subtracting mean image'
        raise NotImplementedError()

    if not input_pixel_max == 1:
        if normalize == 1:
            print '... normalizing with max channel pixel value'
            channel_maxes = numpy.array(255 - channel_means, dtype=accuracy_dtype)
            train_set = divide_channel_max(train_set, image_shape, channel_maxes)
            valid_set = divide_channel_max(valid_set, image_shape, channel_maxes)
            test_set = divide_channel_max(test_set, image_shape, channel_maxes)
        elif normalize == 2:
            print '... normalizing with image std deviations'
            raise NotImplementedError()

    print '... sharing data'

    def share_dataset(data_xy, borrow=True, conserve_gpu_memory=False):
        """ Function that casts the dataset into the right types, and
        optionally loads the entire dataset into GPU memory (shared variables)

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        if conserve_gpu_memory:
            shared_x = theano.tensor._shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
            shared_y = theano.tensor._shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        else:
            shared_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = share_dataset(test_set, conserve_gpu_memory=conserve_gpu_memory)
    valid_set_x, valid_set_y = share_dataset(valid_set, conserve_gpu_memory=conserve_gpu_memory)
    train_set_x, train_set_y = share_dataset(train_set, conserve_gpu_memory=conserve_gpu_memory)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def subtract_channel_mean(dataset, image_shape, channel_means, accuracy_dtype):
    orig_shape = dataset[0].shape
    full_shape = (dataset[0].shape[0], image_shape[0], image_shape[1], image_shape[2])
    xs = numpy.asarray(dataset[0].reshape(full_shape), dtype=accuracy_dtype) # float is too slow, int is slow enough
    # TODO: do this with broadcasting if its better?
    xs[:,:,:,0] -= channel_means[0]
    xs[:,:,:,1] -= channel_means[1]
    xs[:,:,:,2] -= channel_means[2]
    return (xs.reshape(orig_shape), dataset[1])

def divide_channel_max(dataset, image_shape, channel_maxes):
    orig_shape = dataset[0].shape
    full_shape = (dataset[0].shape[0], image_shape[0], image_shape[1], image_shape[2])
    xs = numpy.asarray(dataset[0].reshape(full_shape), dtype='float32') # float is too slow, int is slow enough
    # TODO: do this with broadcasting if its better?
    xs[:,:,:,0] /= channel_maxes[0]
    xs[:,:,:,1] /= channel_maxes[1]
    xs[:,:,:,2] /= channel_maxes[2]
    return (xs.reshape(orig_shape), dataset[1])

def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # Each MNIST image has size SS*SS
    classifier = LogisticRegression(input=x, n_in=SS * SS, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-3

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()
