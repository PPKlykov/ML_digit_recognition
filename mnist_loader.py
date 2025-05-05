import gzip
import pickle
import numpy as np
from PIL import Image
from config import IMAGE_SIZE

def load_data():
    path = '/Users/ppklykov/Desktop/Machine Learning/MNIST/mnist.pkl.gz'
    with gzip.open(path, 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return training_data, validation_data, test_data

def upscale_to_size(flat_image):
    image_28 = flat_image.reshape((28, 28)) * 255.0
    pil_image = Image.fromarray(image_28.astype(np.uint8))
    pil_image = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    image_resized = np.asarray(pil_image) / 255.0
    return image_resized.reshape((IMAGE_SIZE * IMAGE_SIZE, 1))

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [upscale_to_size(x) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [upscale_to_size(x) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [upscale_to_size(x) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e