# -*- coding: utf-8 -*-
"""
Architecture, Hyper-params: http://arxiv.org/abs/1202.2745
Width Normalization, Elastic distortion Hyper-params: http://arxiv.org/abs/1103.4487
Elastic Distortion methodology: http://research.microsoft.com/pubs/68920/icdar03.pdf
"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer

from theanet.theanet.layer.layer import Layer
from theanet.theanet.layer.inlayers import ElasticLayer

import cPickle
import collections

import pdb

SIGMA = 8 # increased from too-extreme seeming 6 in ICDAR 2011
ALPHA = 36

def save_model(name, theano_params, params):
    """
    Will need to load last layer W,b to first layer W,b
    """
    f = open('./models/'+name+'.pkl', 'wb')

    cPickle.dump([param.get_value(borrow=True) for param in theano_params], f, -1)
    cPickle.dump(params, f, -1)
    f.close()

class Ciresan2012Column(object):
    def __init__(self, datasets,
                 nkerns=[32, 48], batch_size=1000, normalized_width=20, distortion=0, cuda_convnet=1,
                 params=[None, None, None, None, None, None, None, None]):
        """ Demonstrates Ciresan 2012 on MNIST dataset

        Some minor differences here:
        ---
        - Ciresan initializes Conv layers with: "uniform random distribution
            in the range [−0.05, 0.05]." (Ciresan IJCAI 2011)
        - Ciresan uses a sigma of 6
        - Ciresan uses nkerns=[20, 40] which were increased here to be nkerns=[32, 48]
            in order to be compatible with cuda_convnet

        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
                              gradient)

        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer

        :type dataset: string
        :param dataset: path to the dataset used for training /testing (MNIST here)

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        :type params: list of None or Numpy matricies/arrays
        :param params: W/b weights in the order: layer3W, layer3b, layer2W, layer2b, layer1W, layer1b, layer0W, layer0b
        """

        layer3W, layer3b, layer2W, layer2b, layer1W, layer1b, layer0W, layer0b = params
        rng = numpy.random.RandomState(23455)

        # TODO: could make this a theano sym variable to abstract
        # loaded data from column instantiation
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # TODO: could move this to train method
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= batch_size
        self.n_valid_batches /= batch_size
        self.n_test_batches /= batch_size

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        learning_rate = T.fscalar()

        # start-snippet-1
        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the column'

        if distortion:
            distortion_layer = ElasticLayer(
                x.reshape((batch_size, 29, 29)),
                29,
                magnitude=ALPHA,
                sigma=SIGMA
            )

            network_input = distortion_layer.output.reshape((batch_size, 1, 29, 29))
        else:
            network_input = x.reshape((batch_size, 1, 29, 29))

        if cuda_convnet:
            layer0_input = network_input.dimshuffle(1, 2, 3, 0)
        else:
            layer0_input = network_input

        layer0_imageshape = (1, 29, 29, batch_size) if cuda_convnet else (batch_size, 1, 29, 29)
        layer0_filtershape = (1, 4, 4, nkerns[0]) if cuda_convnet else (nkerns[0], 1, 4, 4)

        layer0 = LeNetConvPoolLayer(
            rng,
            input=layer0_input,
            image_shape=layer0_imageshape,
            filter_shape=layer0_filtershape,
            poolsize=(2, 2),
            cuda_convnet=cuda_convnet,
            W=layer0W,
            b=layer0b
        )

        layer1_imageshape = (nkerns[0], 13, 13, batch_size) if cuda_convnet else (batch_size, nkerns[0], 13, 13)
        layer1_filtershape = (nkerns[0], 5, 5, nkerns[1]) if cuda_convnet else (nkerns[1], nkerns[0], 5, 5)

        layer1 = LeNetConvPoolLayer(
            rng,
            input=layer0.output,
            image_shape=layer1_imageshape,
            filter_shape=layer1_filtershape,
            poolsize=(3, 3),
            cuda_convnet=cuda_convnet,
            W=layer1W,
            b=layer1b
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        if cuda_convnet:
            layer2_input = layer1.output.dimshuffle(3, 0, 1, 2).flatten(2)
        else:
            layer2_input = layer1.output.flatten(2)

        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in=nkerns[1] * 3 * 3,
            n_out=150,
            W=layer2W,
            b=layer2b,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(
            input=layer2.output,
            n_in=150,
            n_out=10,
            W=layer3W,
            b=layer3b
        )

        # the cost we minimize during training is the NLL of the model
        cost = layer3.negative_log_likelihood(y)

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a function to compute probabilities of all output classes
        self.test_output_batch = theano.function(
            [index],
            layer3.p_y_given_x,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.validate_model = theano.function(
            [index],
            layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        self.params = layer3.params + layer2.params + layer1.params + layer0.params
        self.column_params = [nkerns, batch_size, normalized_width, distortion, cuda_convnet]

        # create a list of gradients for all model parameters
        grads = T.grad(cost, self.params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, grads)
        ]

        self.train_model = theano.function(
            [index, learning_rate],
            cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

    def test_outputs(self):
        test_losses = [
            self.test_output_batch(i)
            for i in xrange(self.n_test_batches)
        ]
        return numpy.concatenate(test_losses)

    def train_column(self, init_learning_rate, n_epochs):
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            cur_learning_rate = numpy.array(init_learning_rate * 0.993**epoch, dtype=numpy.float32)
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):

                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter

                cost_ij = self.train_model(minibatch_index, cur_learning_rate)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in xrange(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in xrange(self.n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print('Optimization complete.')
        nkerns, batch_size, normalized_width, distortion, cuda_convnet = self.column_params
        name = 'ciresan2012_bs%i_nw%i_d%i_%iLayers_cc%i_t%i' % (batch_size, normalized_width, distortion, len(self.params) / 2, cuda_convnet, int(time.time()))
        print('Saving Model as "%s"...' % name)
        save_model(name, self.params, self.column_params)
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

def train_ciresan2012(init_learning_rate=0.001, n_epochs=800,
                         dataset='mnist.pkl.gz',
                         nkerns=[32, 48], batch_size=1000, normalized_width=20, distortion=0, cuda_convnet=1):
    """ Demonstrates Ciresan 2012 on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    datasets = load_data(dataset, normalized_width, 29)
    column = Ciresan2012Column(datasets, nkerns, batch_size, normalized_width, distortion, cuda_convnet)
    column.train_column(init_learning_rate, n_epochs)


if __name__ == '__main__':
    # Should be trained 5 times per digit width normalization (10, 12, 14, 16, 18, 20)
    arg_names = ['command', 'batch_size', 'normalized_width', 'distortion', 'cuda_convnet', 'n_epochs']
    arg = dict(zip(arg_names, sys.argv))

    batch_size = int(arg.get('batch_size') or 100)
    normalized_width = int(arg.get('normalized_width') or 0)
    distortion = int(arg.get('distortion') or 0)
    cuda_convnet = int(arg.get('cuda_convnet') or 0)
    n_epochs = int(arg.get('n_epochs') or 800) # useful to change to 1 for a quick test run

    train_ciresan2012(batch_size=batch_size, normalized_width=normalized_width,
                         distortion=distortion, n_epochs=n_epochs, cuda_convnet=cuda_convnet)
