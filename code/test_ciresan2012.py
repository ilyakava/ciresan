import sys

import cPickle
import numpy
import theano
import theano.tensor as T

from logistic_sgd import load_data
from ciresan2012 import Ciresan2012Column

import pdb

def test_columns(exclude_mode, models, dataset='mnist.pkl.gz'):
    # create data hash that will be filled with data from different normalizations
    all_datasets = {}
    # instantiate multiple columns
    columns = []
    for model in models:
        # load model params
        f = open('./models/'+model)
        params = cPickle.load(f)
        nkerns, batch_size, normalized_width, distortion, cuda_convnet = cPickle.load(f)
        if all_datasets.get(normalized_width):
            datasets = all_datasets[normalized_width]
        else:
            datasets = load_data(dataset, normalized_width, 29)
            all_datasets[normalized_width] = datasets
        # no distortion during testing
        columns.append(Ciresan2012Column(datasets, nkerns, batch_size, normalized_width, 0, cuda_convnet, params))
    print '... Testing columns'
    # call test on all of them recieving 10 outputs
    model_outputs = [column.test_outputs() for column in columns]
    # average 10 outputs
    avg_output = numpy.mean(model_outputs, axis=0)
    # argmax over them
    predictions = numpy.argmax(avg_output, axis=1)
    # output labels and acc
    pred = T.ivector('pred')

    all_true_labels_length = theano.function([], all_datasets.values()[0][2][1].shape)
    remainder = all_true_labels_length() - len(predictions)
    if exclude_mode and remainder:
        print '... Excluding FIRST %i points' % remainder
        true_labels = all_datasets.values()[0][2][1][remainder:]
    elif remainder:
        print '... Excluding LAST %i points' % remainder
        true_labels = all_datasets.values()[0][2][1][:len(predictions)]
    else:
        true_labels = all_datasets.values()[0][2][1][:]

    error = theano.function([pred], T.mean(T.neq(pred, true_labels)))
    acc = error(predictions.astype(dtype=numpy.int32))
    print 'Error across %i columns: %f %%' % (len(models), 100*acc)
    return [predictions, acc]

if __name__ == '__main__':
    # example:
    # python code/test_ciresan2012.py 0 ciresan2012_bs5000_nw16_d1_4Layers_cc0.pkl ciresan2012_bs5000_nw18_d1_4Layers_cc0.pkl
    assert len(sys.argv) > 2
    models = sys.argv[2:]
    test_columns(int(sys.argv[1]), models)
