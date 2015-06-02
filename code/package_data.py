
def package_data(directory, datasets, outfile_path):
    """
    :type directory: string
    :param directory: directory of image data

    :type datasets: tuple of dictionaries, string -> Any
    :param datasets: dictionaries should be in the order
        training, validation, test. The key is the image
        filename and the value is the true label
    """
    assert(len(datasets) == 3)
    print 'done'
