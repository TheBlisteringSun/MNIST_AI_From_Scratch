import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from Read_MNIST_Dataset import MnistDataloader
from os.path import join
import time

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


def Initializing():
    if input("Do you want to upload your weights and biases? [Y]\n").upper() == "Y":
        wbz = np.load(input("What is the file path to the .npz file with weights and biases?\n"))
        w1 = wbz["w1"]
        b1 = wbz["b1"]
        w2 = wbz["w2"]
        b2 = wbz["b2"]
        w3 = wbz["w3"]
        b3 = wbz["b3"]
    else:
        # Initializing weights & biases
        w1 = np.random.rand(20, 784)
        b1 = np.random.rand(20)
        w2 = np.random.rand(20, 20)
        b2 = np.random.rand(20)
        w3 = np.random.rand(10, 20)
        b3 = np.random.rand(10)
    return w1, b1, w2, b2, w3, b3


def Initializing_Gradient(shapes):
    return [np.zeros(shape) for shape in shapes]


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


def LeakyReLu(array):
    output = np.copy(array)
    output[array < 0] *= 0.01
    return output


def dLeakyReLu(array):
    derivatives = np.ones(array.shape)
    derivatives[array < 0] = 0.01
    return derivatives


def FeedForwards_BackwardsPropogation(training_indexes, w1, b1, w2, b2, w3, b3, gradient):
    cost = np.zeros((10,))

    for index in training_indexes:
        # Feed Forward
        input_layer = x_test[index]
        expected_output = np.zeros(10)
        expected_output[y_test[index]] = 1  # One-Hot Encoding

        a1 = LeakyReLu(np.dot(w1, input_layer) + b1)
        a2 = LeakyReLu(np.dot(w2, a1) + b2)
        a3 = LeakyReLu(np.dot(w3, a2) + b3)
        P = a3 / sum(a3)  # Percentage
        C = (P - expected_output) ** 2

        # Back Prop
        # q#, v#, p# are the partial derivatives of C/a#, C/w#, C/b# respectively

        dCP = 2 * (P - expected_output)
        k = [sum(a3) - aj for aj in a3]
        q3 = np.array(list(map(lambda kj: kj / (sum(a3) + kj) ** 2, k))) * dCP  # dC/da3
        v3 = np.array([i * a2 for i in q3]) * dLeakyReLu(a3).reshape((a3.shape[0], 1))
        p3 = np.copy(q3) * dLeakyReLu(a3)

        q2 = np.dot(np.transpose(w3), q3 * dLeakyReLu(a3))
        v2 = np.array([i * a1 for i in q2]) * dLeakyReLu(a2).reshape((a2.shape[0], 1))
        p2 = np.copy(q2) * dLeakyReLu(a2)

        q1 = np.dot(np.transpose(w2), q2 * dLeakyReLu(a2))
        v1 = np.array([i * x_test[index] for i in q1]) * dLeakyReLu(a1).reshape((a1.shape[0], 1))
        p1 = np.copy(q1) * dLeakyReLu(a1)

        for w_or_b, g in zip([v1, p1, v2, p2, v3, p3], gradient):
            g += w_or_b

        cost += C
    return sum(cost), P


def Shuffling_Dataset(datalen, chunk_size):  # Returns chunks of random indexes
    chunks = []
    indexes = [i for i in range(datalen)]
    random.shuffle(indexes)
    for i in range(datalen // chunk_size):
        chunks.append(indexes[chunk_size * i:chunk_size * (i + 1)])
    if datalen % chunk_size != 0:
        chunks.append(indexes[-datalen % chunk_size:])
    return chunks


def Gradient_Decent(l_r, datalen, w1, b1, w2, b2, w3, b3, gradient):
    updated_weights_and_biases = []
    for w_or_b, g in zip([w1, b1, w2, b2, w3, b3], gradient):
        updated_weights_and_biases.append(w_or_b - l_r * g / datalen) # dividing by datalen makes sure the gradient is averaged

    return updated_weights_and_biases

# Having a learning rate < -1 means you might overshoot on max batch_size
# You should have a lower learning rate for smaller batch_sizes
# Having smaller batch_sizes speeds up the learning
# -3e2 good for 10000, -6.9e-1 for 1000
learning_rate = 5.0e-2  # 2.088e-4
num_epochs = 100
batch_size = 10000
costs = []
plt.ion()

fig, ax = plt.subplots()
ln, = ax.plot(costs)

def Minimize_Cost(learning_rate, num_epochs, batch_size):
    global lowest_cost_params
    global current_params
    global history
    global costs
    w1, b1, w2, b2, w3, b3 = Initializing()  # Initializing weights and biases
    for epoch in range(num_epochs):
        training_sets = Shuffling_Dataset(len(y_test), batch_size)
        for i in range(len(y_test) // batch_size):
            # Initializing the gradient
            gradient = Initializing_Gradient(
                [w1.shape, b1.shape, w2.shape, b2.shape, w3.shape, b3.shape])

            cost, history = FeedForwards_BackwardsPropogation(training_sets[i], w1, b1, w2, b2, w3, b3, gradient)
            w1, b1, w2, b2, w3, b3 = Gradient_Decent(learning_rate, batch_size, w1, b1, w2, b2, w3, b3, gradient)
            cost /= batch_size
            costs.append(cost)

            if cost == min(costs):
                lowest_cost_params = w1, b1, w2, b2, w3, b3

        if epoch % max(num_epochs // 20, 1) == 0:  # Progress Percentage
            print(f"Progress: {round(epoch / num_epochs, 2) * 100}%")
            plt.cla()
            plt.plot(costs)
            plt.show()
    current_params = w1, b1, w2, b2, w3, b3

plt.plot([cost / batch_size for cost in costs if cost < batch_size])
plt.show()
# gen_costs = Minimize_Cost(learning_rate, num_epochs, batch_size)
# last_cost = next(gen_costs)
# animation = FuncAnimation(plt.gcf(), func=update_plot, interval=1000,save_count=10)

Minimize_Cost(learning_rate, num_epochs, batch_size)

print(history)
print(f"{costs[0] = }, {costs[-1] = }, {min(costs) = }")

if input("Save? [Y]\n").upper() == "Y":
    if input("Save the weights & biases with the lowest cost? [L] (Otherwise saves current state)\n").upper() == "L":
        w1, b1, w2, b2, w3, b3 = lowest_cost_params
    else:
        w1, b1, w2, b2, w3, b3 = current_params
    np.savez(join(input("Input path to where to save the weights and biases:\n"), "Weights_and_Biases"), w1=w1, b1=b1,
             w2=w2, b2=b2, w3=w3, b3=b3)
