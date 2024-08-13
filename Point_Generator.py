import math
import random
from matplotlib import pyplot as plt
import numpy as np


num_params = int(input("how many params"))

params = []
for i in range(num_params):
    params.append(random.randint(-10 // (i+1), 10 // (i+1))/3)

xs = random.sample(range(-20,20), int(input("how many points")))


points = []
for x in xs:
    y = sum([param * math.pow(x, i) for i, param in enumerate(params)])
    points.append((x, y))

print(num_params, params, points)
for x, y in points:
    print(f"{x}, {y}")

xs = np.linspace(-100, 100, 400)
ys = sum([param * np.power(xs, i) for i, param in enumerate(params)])

plt.plot(xs, ys)
plt.scatter(*zip(*points))
plt.show()

