# encoding=utf8
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import scipy

# Load the data set
X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"), encoding='latin1')

img = scipy.misc.toimage(X[0])

img.save('pkl_image01.bmp')