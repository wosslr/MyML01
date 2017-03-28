import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import datasets
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

def show_digit(data):
    pixels = np.array(data, dtype='uint8')
    pixels = pixels.reshape(28, 28)
    plt.imshow(pixels, cmap=plt.cm.gray)
    plt.show()

def show_top_100(data):
    fig, axes = plt.subplots(10, 10)
    i = 0
    for ax in axes.ravel():
        pixels = np.array(data[i], dtype='uint8')
        pixels = pixels.reshape(28, 28)
        ax.matshow(pixels, cmap=plt.cm.gray)
        i += 1
    plt.show()

def read_image_as_arr(file_name):
    img = Image.open(file_name)
    gray_img = img.convert("L")
    im_array = np.array(gray_img)
    im_array = im_array.ravel()
    return im_array

MAX_TRAIN_SIZE = 31000
digits = datasets.load_digits()

mnist = datasets.fetch_mldata('MNIST original', data_home='datahome')

clf = svm.SVC(gamma=0.0001, C=100)

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

# x,y = mnist.data[:-10], mnist.target[:-10]

X, Y = shuffle(mnist.data, mnist.target)

x, y = X / 255., Y

# pixels = np.array(mnist.data[0], dtype='uint8')
# pixels = pixels.reshape(28, 28)
# plt.imshow(pixels, cmap=plt.cm.gray)
# plt.show()

mlp.fit(x[100:MAX_TRAIN_SIZE], y[100:MAX_TRAIN_SIZE])

show_top_100(X[:100])

while True:
    # index = input()
    # index = int(index)
    # pre_data = x[index]
    # predication = mlp.predict([pre_data])
    # print(predication)
    # show_digit(X[index])
    test_image = input()
    test_data = read_image_as_arr(test_image)
    predication = mlp.predict([test_data])
    print(predication[0])

# mlp.fit(x_train, y_train)
#
# print("Training set score: %f" % mlp.score(x_train, y_train))
# print("Test set score: %f" % mlp.score(x_test, y_test))
#
# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())
#
# plt.show()