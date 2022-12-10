import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    s = np.copy(x)
    for i in range(x.shape[0]):
        s[i, 0] = 1 / (1 + np.exp(-x[i, 0]))
    return s


def derivative_sigmoid(x):
    s = np.copy(x)
    for i in range(x.shape[0]):
        d = s[i, 0]
        s[i, 0] = d * (1 - d)
    return s


class Network:
    def __init__(self, structure):
        self.structure = structure
        self.num_layers = len(structure)
        self.B = [np.zeros((l, 1)) for l in structure[1:]]
        self.W = [
            np.zeros((l, next_l)) for l, next_l in zip(structure[:-1], structure[1:])
        ]


def forward(model, train):
    z_list = []
    a_list = []
    for i in range(model.num_layers - 1):
        if i == 0:
            z_i = np.matmul(np.transpose(model.W[i]), train) + model.B[i]
            z_list.append(z_i)
            a_i = sigmoid(z_i)
            a_list.append(a_i)
        else:
            z_i = np.matmul(np.transpose(model.W[i]), a_list[i - 1]) + model.B[i]
            z_list.append(z_i)
            a_i = sigmoid(z_i)
            a_list.append(a_i)
            if i == model.num_layers - 2:
                a_list[i] = z_i
                y = z_i
    return (y, a_list)


def backward(model, train, y_real):
    y, a_list = forward(model, train)
    deltas = []
    w_deriv = []
    b_deriv = []
    count = -1
    for m in range(model.num_layers - 1, 0, -1):
        count += 1
        if m == model.num_layers - 1:
            delta_y = y - y_real
            deltas.append(delta_y)
            d1 = a_list[m - 2]
            d2 = delta_y
            deriv_w = np.multiply(a_list[m - 2], delta_y)
            w_deriv.append(deriv_w)
            b_deriv.append(delta_y)
        elif m == model.num_layers - 2:
            d1 = deltas[count - 1]
            d2 = model.W[m]
            d3 = derivative_sigmoid(a_list[m - 1])
            delta = np.multiply(np.multiply(d2, d3), d1)
            deltas.append(delta)
            d4 = a_list[m - 2]
            deriv_w = np.matmul(d4, delta.transpose())
            w_deriv.append(deriv_w)
            b_deriv.append(delta)
        elif m != 1:
            d1 = deltas[count - 1]
            d2 = model.W[m]
            d3 = derivative_sigmoid(a_list[m - 1])
            d4 = np.multiply(d2, d3)
            delta = np.multiply(d4, d1)
            deltas.append(delta)
            b_deriv.append(delta)
            d5 = a_list[m - 2]
            deriv_w = np.matmul(d5, delta.transpose())
            w_deriv.append(deriv_w)
        else:
            d1 = deltas[count - 1]
            d2 = model.W[m]
            d3 = derivative_sigmoid(a_list[m - 1])
            d4 = np.multiply(d3, d2)
            delta = np.matmul(d4, d1)
            deltas.append(delta)
            b_deriv.append(delta)
            deriv_w = np.matmul(train, delta.transpose())
            w_deriv.append(deriv_w)
    return w_deriv, b_deriv


def SVM_SGD(df, model, epoch, gamma_0, d):
    objective = []
    for t in range(0, epoch):
        gamma_t = gamma_0 / (1 + (gamma_0 / d) * t)
        df_shuffle = df.sample(frac=1)
        df_shuffle = df_shuffle.reset_index(drop=True)
        N = df_shuffle.shape[0]
        for row in range(0, df_shuffle.shape[0]):
            x_i = np.reshape(df_shuffle.iloc[row, :-1].to_numpy(), (input_width, 1))
            y_i = df_shuffle.iloc[row, -1]
            deriv_w_list, deriv_b_list = backward(model, x_i, y_i)
            N = model.num_layers - 2
            for layer in range(len(deriv_w_list)):
                model.W[N - layer] = model.W[N - layer] - np.multiply(
                    deriv_w_list[layer], gamma_t
                )
                model.B[N - layer] = model.B[N - layer] - np.multiply(
                    deriv_b_list[layer], gamma_t
                )
            w_final = model.W
            b_final = model.B
        objective.append(loss(df, model))
    return w_final, b_final, objective


def loss(df, model):
    loss_sum = 0
    for row in range(0, df.shape[0]):
        predict = 0
        x_i = np.reshape(df.iloc[row, :-1].to_numpy(), (input_width, 1))
        y_i = df.iloc[row, -1]
        predict, _ = forward(model, x_i)
        loss = 0.5 * (predict[0][0] - y_i) ** 2
        loss_sum = loss_sum + loss
    return loss_sum


def error(df, model):
    error = 0
    for row in range(0, df.shape[0]):
        x_i = np.reshape(df.iloc[row, :-1].to_numpy(), (input_width, 1))
        y_i = df.iloc[row, -1]
        y_predict, a_list = forward(model, x_i)
        predict = np.sign(y_predict[0][0])
        if predict == 0:
            predict = 1
        if predict != (y_i):
            error = error + 1
    return error / df.shape[0]


columns = ["variance", "skewness", "curtosis", "entropy", "label"]

# Reading training and testing data
train = pd.read_csv(r'nn\train.csv', names=columns, dtype=np.float64())
train = train.replace({"label": 0}, -1)
test = pd.read_csv(r"nn\test.csv", names=columns, dtype=np.float64())
test = test.replace({"label": 0}, -1)

# Setting hyperparameter
print("input epoch value")
epoch = int(input())
print("input Gamma_0 value")
Gamma_0 = float(input())
print("input d value")
d = float(input())

input_width = 4
output_width = 1
widths = [5, 10, 25, 50, 100]
for hidden_width in widths:
    model = Network([input_width, hidden_width - 1, hidden_width - 1, output_width])
    w_initial = model.W
    b_initial = model.B
    w, b, objective = SVM_SGD(train, model, epoch, Gamma_0, d)
    plt.plot(objective)
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
    train_error = error(train, model)
    test_error = error(test, model)
    print("epoch=", epoch, "Gamma_0=", Gamma_0, "d=", d, "hidden_width", hidden_width)
    print("train_error", train_error)
    print("test_error", test_error, "\n")
