import pickle
import gzip
import numpy as np

def data_loader():
	with gzip.open('mnist.pkl.gz','rb') as ff :
	    u = pickle._Unpickler( ff )
	    u.encoding = 'latin1'
	    training_data, validation_data, test_data = u.load()
	return (training_data, validation_data, test_data)

def load_data_wrapper():
	tr_d, va_d, te_d = data_loader()
	training_inputs = [np.reshape(x, (784,1)) for x in tr_d[0]]
	training_results = [vectorised_result(y) for y in tr_d[1]]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
	validation_data = zip(validation_inputs, va_d[1])
	test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
	test_data = zip(test_inputs, te_d[1])
	return (training_data, validation_data, test_data)

def vectorised_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e