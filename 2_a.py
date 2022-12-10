import numpy as np


def sigmoid(x):
    s = np.copy(x)
    n = x.shape[0]
    for i in range(n):
        s[i, 0] = 1 / (1 + np.exp(-x[i, 0]))
    return s


def sigmoid_derivative(x):
    s = np.copy(x)
    n = x.shape[0]
    for i in range(n):
        d = s[i, 0]
        s[i, 0] = d * (1 - d)
    return s


class Network:
    def __init__(self, model):
        self.num_layers = len(model)
        self.B = [np.array([[-1], [1]]), np.array([[-1], [1]]), np.array([-1])]
        self.W = [
            np.array([[-2, 2], [-3, 3]]),
            np.array([[-2, 2], [-3, 3]]),
            np.array([[2], [-1.5]]),
        ]


# Setting hyperparamters
train_dataset = np.ones((2, 1))
model_structure = [2, 2, 2, 1]
model = Network(model_structure)
y_real = 1

# Forward Propagation
list_z = []
list_a = []
for i in range(model.num_layers - 1):
    if i == 0:
        z1 = np.transpose(model.W[i])
        z2 = train_dataset
        z3 = np.matmul(np.transpose(model.W[i]), train_dataset)
        z4 = model.B[i]
        z_i = z4 + z3
        list_z.append(z_i)
        a_i = sigmoid(z_i)
        list_a.append(a_i)
    else:
        z1 = np.transpose(model.W[i])
        z2 = list_a[i - 1]
        z3 = np.matmul(np.transpose(model.W[i]), list_a[i - 1])
        z4 = model.B[i]
        z_i = z3 + z4
        list_z.append(z_i)
        a_i = sigmoid(z_i)
        list_a.append(a_i)

        if i == model.num_layers - 2:
            list_a[i] = z_i
            y_hat = z_i


# Backpropagation
gamma = 1
deltas = []
w_deriv = []
b_deriv = []
h_prime = []
count = -1
for m in range(model.num_layers - 1, 0, -1):
    count += 1
    if m == model.num_layers - 1:
        delta_y = y_hat - y_real
        deltas.append(delta_y)
        d1 = list_a[m - 2]
        d2 = delta_y
        deriv_w = np.multiply(list_a[m - 2], delta_y)
        w_deriv.append(deriv_w)
        b_deriv.append(delta_y)
    elif m == model.num_layers - 2:
        d1 = deltas[count - 1]
        d2 = model.W[m]
        d3 = sigmoid_derivative(list_a[m - 1])
        delta = np.multiply(np.multiply(d1, d2), d3)
        deltas.append(delta)
        d4 = list_a[m - 2]
        deriv_w = np.matmul(d4, delta.transpose())
        w_deriv.append(deriv_w)
        b_deriv.append(delta)
    else:
        d1 = deltas[count - 1]
        d2 = model.W[m]
        d3 = sigmoid_derivative(list_a[m - 1])
        d4 = np.matmul(d2, d1)
        delta = np.multiply(d4, d3)
        deltas.append(delta)
        b_deriv.append(delta)
        if m != 1:
            d5 = list_a[m - 2]
            deriv_w = np.matmul(d5, delta.transpose())
            w_deriv.append(deriv_w)
        else:
            deriv_w = np.matmul(train_dataset, delta.transpose())
            w_deriv.append(deriv_w)

n = model.num_layers - 2
for layer in range(len(w_deriv)):
    m1 = model.W[n - layer]
    m2 = np.multiply(w_deriv[layer], gamma)
    m3 = m1 - m2
    m4 = model.B[n - layer]
    m5 = np.multiply(b_deriv[layer], gamma)
    m6 = m4 - m5
    model.W[n - layer] = model.W[n - layer] - np.multiply(w_deriv[layer], gamma)
    model.B[n - layer] = model.B[n - layer] - np.multiply(b_deriv[layer], gamma)
    print("layer w =", n + 1 - layer, "\n", w_deriv[layer])
    print("layer b =", n + 1 - layer, "\n", b_deriv[layer], "\n")
