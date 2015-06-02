# -*- coding: utf-8 -*-
from PIL import Image
import numpy
import scipy.misc
import gzip
from nyanbar import NyanBar

def package_data(directory, datasets, image_shape, outfile_path):
    """
    Outputs a *.npz file of image data in 3 sets

    :type directory: string
    :param directory: directory of image data

    :type datasets: tuple of dictionaries, string -> Any
    :param datasets: dictionaries should be in the order
        training, validation, test. The key is the image
        filename and the value is the true label

    :type image_shape: tuple of ints

    :type outfile_path: string
    """
    assert(len(datasets) == 3)

    xsize = numpy.prod(image_shape)

    x_datasets = [numpy.zeros((len(dataset), xsize), dtype=numpy.uint8) for dataset in datasets]
    y_datasets = [numpy.array(dataset.values(), dtype=numpy.uint8) for dataset in datasets]

    print "| " + ("⚐ ⚑ " * 19) + "-|"
    pb = NyanBar(tasks=sum([len(dataset) for dataset in datasets]))
    for j, dataset in enumerate(datasets):
        for i, image_name in enumerate(dataset.keys()):
            pb.task_done()
            im = Image.open(directory + image_name)
            x_datasets[j][i, :] = numpy.array(im.getdata(), dtype=numpy.uint8).reshape(xsize)
    pb.finish()

    print '... saving data'
    # cPickle too slow (takes many minutes for tens of thousands of images over 100x100x3)
    saveme = [x_datasets[0], y_datasets[0], x_datasets[1], y_datasets[1], x_datasets[2], y_datasets[2]]
    numpy.savez(open(outfile_path, 'wb'), *saveme)

    print 'done'
