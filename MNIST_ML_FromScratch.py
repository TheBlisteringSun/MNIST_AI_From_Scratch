import random
import matplotlib.pyplot as plt
import numpy as np
from Read_MNIST_Dataset import MnistDataloader
from os.path import join

#
# Set file paths based on added MNIST Datasets
#
input_path = input("What is the file path to the MNIST Dataset?\n")
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_train = np.stack(x_train)
y_train = np.stack(y_train)
x_test = np.stack(x_test)
y_test = np.stack(y_test)

x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

#
# Initializing weights & biasis
#

w1 = np.random.rand(20, 784)
b1 = np.random.rand(20)
w2 = np.random.rand(20, 20)
b2 = np.random.rand(20)
w3 = np.random.rand(10, 20)
b3 = np.random.rand(10)

# Gradient Initialization

gw1 = np.zeros((20, 784))
gb1 = np.zeros((20,))
gw2 = np.zeros((20, 20))
gb2 = np.zeros((20,))
gw3 = np.zeros((10, 20))
gb3 = np.zeros((10,))


def ReLu(array):
    return np.array(list(map(lambda x: max(0, x), array)))


def dReLu(array, a):
    out = []
    for i, value in enumerate(array):
        if a[i] == 0:
            out.append(np.zeros(value.shape))
        else:
            out.append(value)
    return np.array(out)


def Forward_Propogation(training_indexes, w1, b1, w2, b2, w3, b3):  # Forward Prop
    a1s, a2s, a3s, Ps = [], [], [], []
    for index in training_indexes:
        input_layer, number = x_test[index], y_test[index]
        a1s.append(ReLu(np.dot(w1, input_layer) + b1))
        a2s.append(ReLu(np.dot(w2, a1s[-1]) + b2))
        a3s.append(ReLu(np.dot(w3, a2s[-1]) + b3))
        Ps.append(a3s[-1] / sum(a3s[-1]))  # Percentage
    return a1s, a2s, a3s, Ps


def Backward_Propagation(training_indexes, w1, b1, a1s, w2, b2, a2s, w3, b3, a3s, Ps):
    # Back Prop
    # q#, v#, p# are the partial derivatives of a#, w#, b# respectively
    cost = np.zeros((10,))
    for a_index, index in enumerate(training_indexes):
        expected_output = np.zeros((10))
        expected_output[y_test[index]] = 1  # One-Hot Encoding
        squarer = lambda x: x ** 2

        dPs = 2 * (Ps[a_index] - expected_output)
        q3 = np.array(list(map(lambda x: sum(a3s[a_index] - x) / sum(a3s[a_index]) ** 2, a3s[a_index]))) * dPs  # dC/da3
        dead3, dead2, dead1 = np.where(a3s[a_index] == 0), np.where(a2s[a_index] == 0), np.where(a1s[a_index] == 0)

        v3, p3 = np.array([i * a2s[a_index] for i in q3]), np.copy(q3)
        q2 = np.dot(np.transpose(w3), dReLu(q3, a3s[a_index]))

        v2, p2 = np.array([i * a1s[a_index] for i in q2]), np.copy(q2)
        q1 = np.dot(np.transpose(w2), dReLu(q2, a2s[a_index]))

        v1, p1 = np.array([i * x_test[index] for i in q1]), np.copy(q1)

        v3[dead3] = np.zeros(20)
        v2[dead2] = np.zeros(20)
        v1[dead1] = np.zeros(784)

        np.add(gw1, v1, out=gw1)
        np.add(gb1, p1, out=gb1)
        np.add(gw2, v2, out=gw2)
        np.add(gb2, p2, out=gb2)
        np.add(gw3, v3, out=gw3)
        np.add(gb3, p3, out=gb3)

        cost += squarer(Ps[a_index] - expected_output)
    return sum(cost)


def Shuffling_Dataset(datalen, chunk_size):  # Returns chunks of random indexes
    chunks = []
    indexs = [i for i in range(datalen)]
    random.shuffle(indexs)
    for i in range(datalen // chunk_size):
        chunks.append(indexs[i:i + chunk_size])
    if datalen % chunk_size != 0:
        chunks.append(indexs[-datalen % chunk_size:])
    return chunks


def Gradient_Decent(l_r, datalen, w1, b1, w2, b2, w3, b3, gw1, gb1, gw2, gb2, gw3, gb3):
    nw1 = np.add(gw1 * l_r / datalen, w1)
    nb1 = np.add(gb1 * l_r / datalen, b1)
    nw2 = np.add(gw2 * l_r / datalen, w2)
    nb2 = np.add(gb2 * l_r / datalen, b2)
    nw3 = np.add(gw3 * l_r / datalen, w3)
    nb3 = np.add(gb3 * l_r / datalen, b3)

    return nw1, nb1, nw2, nb2, nw3, nb3


learning_rate = -1e-10
costs = []
for _ in range(5):
    training_sets = Shuffling_Dataset(len(y_test), 2000)
    for i in range(len(y_test)//len(training_sets[0])):
        a1s, a2s, a3s, Ps = Forward_Propogation(training_sets[i], w1, b1, w2, b2, w3, b3)
        cost = Backward_Propagation(training_sets[i], w1, b1, a1s, w2, b2, a2s, w3, b3, a3s, Ps)
        w1, b1, w2, b2, w3, b3 = Gradient_Decent(learning_rate, len(training_sets[i]), w1, b1, w2, b2, w3, b3, gw1,
                                                 gb1, gw2, gb2, gw3, gb3)
        costs.append(cost)

print(f"{costs[0] = }, {costs[-1] = }, {min(costs) = }")
# print(w1, b1, w2, b2, w3, b3)
plt.plot(costs)
plt.show()

if input("Save? [Y]\n").upper() == "Y":
    np.savez(join(input("Input path to where to save the weights and biases:\n"), "Weights_and_Biases"), w1=w1, b1=b1,
             w2=w2, b2=b2, w3=w3, b3=b3)