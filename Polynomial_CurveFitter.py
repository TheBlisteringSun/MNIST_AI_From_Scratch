import math
import random
from matplotlib import pyplot as plt
import numpy as np


# This is a 2D curvefitting model that uses Backpropagation and Gradient Decent


def forward_prop(params, x, y):  # returns the residual for a single point
    pred_y = 0
    for i, param in enumerate(params):
        pred_y += math.pow(x, i) * param

    return pred_y - y


def error(params, points):  # Loss func
    loss = 0
    loss_log = []
    for x, y in points:
        residual = forward_prop(params, x, y)
        loss_log.append((x, residual))
        try:
            loss += math.pow(residual, 2)
        except OverflowError:
            loss += 1e30
    loss /= len(points)

    return loss, loss_log


def backwards_prop(params, loss_log):  # returns the partial derivatives of each param
    derivatives = {}

    for i, param in enumerate(params):
        partial_dv_const = 0
        partial_dv_coeff = 0
        for x, residual in loss_log:
            partial_dv_const += math.pow(x, i) * (residual - param * math.pow(x, i))
            partial_dv_coeff += math.pow(x, 2 * i)

        derivatives[i] = (partial_dv_const / len(loss_log), partial_dv_coeff / len(loss_log))

    return derivatives


def gradient_decent(params, derivatives, l_r, p=False, r=0):  # nudges params using the partial derivatives and l_r

    for i, param in enumerate(params):
        target = -derivatives[i][0] / derivatives[i][1]  # the x value where the derivative == 0
        # percentage of the distance to the target and some random value (if enabled)
        nudge = l_r * (param - target) + l_r / 2 * random.random() * r

        # nudge = l_r * (derivatives[i][1] * param + derivatives[i][0])  l_r * derivative
        params[i] -= nudge

        if p is True:  # prints the value of param vs new value of param on the derivative graph
            xs = np.linspace(param - 2 * nudge, param + 2 * nudge, 5)
            ys = derivatives[i][0] + derivatives[i][1] * xs
            plt.plot(xs, ys)
            plt.scatter(param, derivatives[i][0] + derivatives[i][1] * param, 20)

            plt.scatter(params[i], derivatives[i][0] + derivatives[i][1] * params[i], 40)
            plt.show()

    return params


def main():
    num_params, num_points = int(input()), int(input())
    points = []
    for i in range(num_points):
        points.append(tuple(float(num) for num in input().split(", ")))

    print(points)

    if input("do you have params") == "T":
        params = [float(i) for i in input().strip("[]").split(", ")]
    else:
        params = [random.randint(-10 // (i + 1), 10 // (i + 1)) / 3 for i in range(num_params) or 0.2]
    print(params)

    error_log = [(float("inf"), None), (float("inf"), None)]
    l_r = 0.5
    milestone = 20
    for i in range(1, 200000):
        if i >= 16000:  # test to modify l_r
            l_r = max(0.99 / len(params), 0.5)
        loss, loss_log = error(params, points)
        derivatives = backwards_prop(params, loss_log)
        if loss < milestone:
            print(loss)
            # l_r *= 0.99
            milestone *= 0.5

        # code below tries to prevent getting stuck in local minimums
        local_minimum = 0 if error_log[-1][0] - error_log[-2][0] < 0.0005 and 10 < error_log[-1][0] < 150 else 0
        params = gradient_decent(params, derivatives, l_r, i == -1, local_minimum)
        error_log.append((loss, params))

    sorted_params = sorted(error_log, key=lambda x: x[0])

    print(params)
    print(sorted_params[0])
    print(f"Absolute Error:{sum([math.pow(i[1], 2) for i in error(params, points)[1]])}")  # old metric (no average)
    plt.plot(next(zip(*error_log))[150000::1000])
    plt.show()


if __name__ == "__main__":
    main()
