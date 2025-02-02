"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import math
import os
import sys
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import deepxde as dde
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

current_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_path, "../../src/")
if src_path in sys.path:
    sys.path.remove(src_path)
sys.path.insert(0, src_path)
from utilities3 import *
import test_Numerical as tn

dde.config.enable_xla_jit(mode=True)
# List all physical GPUs
gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        # Set TensorFlow to allocate only the necessary video memory space
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Printing exception
        print(e)

logical_gpus = tf.config.list_logical_devices("GPU")
print("Logical GPUs after setting visible devices:", logical_gpus)

# tf.config.set_visible_devices(tf.config.list_physical_devices("GPU")[2], "GPU")


def periodic(x):
    x *= 2 * np.pi
    return tf.concat(
        [tf.math.cos(x), tf.math.sin(x), tf.math.cos(2 * x), tf.math.sin(2 * x)], 1
    )


begin = time.time()
path = "/home/ext8/mengbin/BI-TDONet_update/LBVP/data_interior_Dirichlet_problem"
words = os.path.basename(path).split("_")[-3:]
problem = "".join(word[0] for word in words).upper()
print(problem)
start_time = time.time()

data_path = path + "/data.mat"
print("Data location:", data_path)
data = sio.loadmat(data_path)
para = data["para"]
phi = data["phi"]
f = data["f"]
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Data reading time: {elapsed_time:.4f} seconds")

print(
    f"Shape of boundary dataset: {para.shape}\n",
    f"Shape of density dataset: {phi.shape}\n",
    f"Shape of rhs dataset:{f.shape}",
)


m = f.shape[0]
np.random.seed(1117)
idx = np.array(range(m))
np.random.shuffle(idx)
para = para[idx, :]
phi = phi[idx, :]
f = f[idx, :]

M = 128
m = f.shape[0]
N = (f.shape[1] - 1) // 2

print(para.shape, phi.shape, f.shape)

para_train = para[0 : 8 * m // 10, :]
para_test = para[8 * m // 10 :, :]
phi_train = phi[0 : 8 * m // 10, :]
phi_test = phi[8 * m // 10 :, :]
f_train = f[0 : 8 * m // 10, :]
f_test = f[8 * m // 10 :, :]


# initialization
M = 128
N = (f.shape[1] - 1) // 2

# [0,2pi)
trunk = np.linspace(0, 2 * np.pi, M, endpoint=False)

train_para_x = para_train[:, : 2 * N + 1]
train_para_y = para_train[:, 2 * N + 1 :]
test_para_x = para_test[:, : 2 * N + 1]
test_para_y = para_test[:, 2 * N + 1 :]

# boundary
train_para = np.concatenate(
    [tn.to_point(train_para_x, trunk), tn.to_point(train_para_y, trunk)],
    axis=1,
)
test_para = np.concatenate(
    [tn.to_point(test_para_x, trunk), tn.to_point(test_para_y, trunk)], axis=1
)

# rhs
train_f = tn.to_point(f_train, trunk)
test_f = tn.to_point(f_test, trunk)

# density function
train_phi = tn.to_point(phi_train, trunk)
test_phi = tn.to_point(phi_test, trunk)

trunk = np.reshape(trunk, [-1, 1])

# show data shape
print(
    f"Shape of para_train: {para_train.shape}, "
    f"Shape of f_train: {f_train.shape}, "
    f"Shape of phi_train: {phi_train.shape}"
    f"Shape of trunk: {trunk.shape}, "
)


# packed data
X_train = (train_para, train_f, trunk)
y_train = train_phi
X_test = (test_para, test_f, trunk)
y_test = test_phi

data = dde.data.QuadrupleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

# set net structure
dim_x = 1
Layers = [
    [2 * M, 600, 600, 600, 600],
    [M, 600, 600, 600, 600],
    [dim_x, 600, 600, 600, 600],
]
net = dde.nn.MIONetCartesianProd(
    Layers[0],
    Layers[1],
    Layers[2],
    "relu",
    "Glorot normal",
)
net.apply_feature_transform(periodic)
model = dde.Model(data, net)
name = f"LBVP_BI-DeepONet_{problem}_0111"
# Initialize network
state = "test"
# state = "train"

epochs = 5000
batch_size = 2**13
iterations_per_epoch = math.ceil(train_para.shape[0] / batch_size)
iterations = epochs * iterations_per_epoch

learning_rate = 0.001
lr_decay = 0.5

print("Layers=", Layers)
print(
    "epochs= %d, batchs=%d, iterations=%d, lr= %3f, lr_decay=%3f"
    % (epochs, batch_size, iterations, learning_rate, lr_decay)
)
print(name)
# Compile and Train
model.compile(
    "adam",
    lr=learning_rate,
    loss="mean l2 relative error",
    decay=("inverse time", iterations // 100, lr_decay),
    metrics=["mean l2 relative error"],
)
checkpoint_save_path = "model/%s/%s-%s.ckpt" % (name, name, iterations)
if os.path.exists(checkpoint_save_path + ".index"):
    print("-------------load the model-----------------")
    model.restore(checkpoint_save_path, device=None, verbose=0)

loss_path = "loss/%s.mat" % name
if os.path.exists(loss_path):
    print("-----------------exist-----------------")
# Visualization
plt.rcParams.update(
    {
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{bm}",
        "font.size": 30,
        "font.weight": "bold",
    }
)

if state == "train":
    losshistory, train_state = model.train(
        batch_size=batch_size,
        display_every=iterations_per_epoch,
        iterations=iterations,
    )
    model.save("model/%s/%s" % (name, name), protocol="backend", verbose=0)
    # dde.utils.plot_loss_history(losshistory)
    dde.utils.save_loss_history(losshistory, loss_path)
    train_loss = losshistory.loss_train
    val_loss = losshistory.loss_test
elif state == "test":
    loss = np.loadtxt(loss_path)
    loss = loss[1:, :]
    train_loss = loss[:, 1]
    val_loss = loss[:, 2]

else:
    # If 'state' is neither 'train' nor 'test', raise an error
    raise ValueError(f"Invalid state: {state}. Please choose either 'train' or 'test'.")

plt.figure(figsize=(8, 6))

# Plot training loss
plt.plot(
    np.array(range(len(train_loss))),
    # * 10 * 5 / 4,  # X-axis scaling
    np.reshape(np.log10(train_loss), [-1, 1]),  # Logarithmic Y-axis scaling for loss
    linewidth=5.0,  # Line thickness
    label="BI-DeepONet \n Training loss",  # Label for the legend
)

# Plot testing loss
plt.plot(
    np.array(range(len(val_loss))),
    #   * 10 * 5 / 4,  # X-axis scaling
    np.reshape(np.log10(val_loss), [-1, 1]),  # Logarithmic Y-axis scaling for val_loss
    linewidth=5.0,  # Line thickness
    label="BI-DeepONet \n Testing loss",  # Label for the legend
)

# Set axis properties
ax = plt.gca()  # Get the current axis object
ax.yaxis.set_major_locator(
    MaxNLocator(integer=True)
)  # Ensure Y-axis has integer tick marks

# Set labels and title with bold fonts
plt.xlabel("Epoch", fontweight="bold")
plt.ylabel("lg(MRE)", fontweight="bold")
plt.title("BI-DeepONet Loss", fontweight="bold")

# Display legend
plt.legend()

# Adjust layout for better spacing
plt.tight_layout(pad=0)
plt.savefig(
    f"../Figures/{name}_loss.pdf",
    format="pdf",
    bbox_inches="tight",
)
plt.savefig(
    f"../Figures/{name}_loss.eps",
    format="eps",
    bbox_inches="tight",
)
###################################
############ Testing ##############
###################################

t = np.linspace(0, 2 * np.pi, M, endpoint=False)
t1 = np.linspace(0, 2 * np.pi, M)
m1 = f_test.shape[0]

MAE = []
MRE = []
terror = []

# Divide the test set into 10 groups
NN = 100
Nn = m1 // NN
MAE = []
MRE = []
terror = []


# Main loop for full groups
for i in range(NN):
    a = Nn * i
    b = Nn * (i + 1) if i < NN - 1 else m1  # Adjust b for the last group
    input = (test_para[a:b, :], test_f[a:b, :], trunk)
    begin = time.time()
    phi_predict = model._outputs([], input)
    end = time.time()
    terror.append(end - begin)
    phi_predict_fourier = np.fft.fft(phi_predict) * np.sqrt(2 * np.pi) / M
    phi_predict = tn.resort_fourier(phi_predict_fourier, N)
    error = phi_test[a:b, :] - phi_predict
    MAE.append(np.linalg.norm(error, axis=1))
    MRE.append(np.linalg.norm(error, axis=1) / np.linalg.norm(phi_test[a:b, :], axis=1))


# Flatten MAE and MRE for consistent dimensionality
terror = np.vstack(terror)  # Stack the results along a new axis (vertically)
MAE = np.vstack(MAE)
MRE = np.vstack(MRE)
# Calculate mean and variance
mean_MAE = np.mean(MAE)
mean_MRE = np.mean(MRE)
var_MAE = np.var(MAE)
var_MRE = np.var(MRE)

print("Average MAE = {:.4e}".format(mean_MAE))
print("Average MRE = {:.4e}".format(mean_MRE))
print("Variance of MAE = {:.4e}".format(var_MAE))
print("Variance of MRE = {:.4e}".format(var_MRE))
print("Average inference time = {:.4e} milliseconds".format(np.sum(terror) / m1 * 1000))


###################################
############ Examples ##############
###################################
t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
t1 = np.linspace(0, 2 * np.pi, M)
m1 = f_test.shape[0]
r1 = np.random.randint(0, m1)
r2 = np.random.randint(0, m1)

para = np.reshape(para_test[r1], [1, -1])
phi_true_f = np.reshape(phi_test[r1], [1, -1])
f_f = np.reshape(f_test[r1], [1, -1])
px = np.reshape(para[0, : 2 * N + 1], [1, -1])
py = np.reshape(para[0, 2 * N + 1 :], [1, -1])
x = tn.to_point(px, t)
y = tn.to_point(py, t)
f = tn.to_point(f_f, t)
f_fourier = np.fft.fft(f) * np.sqrt(2 * np.pi) / M
x = np.reshape(x, [1, -1])
y = np.reshape(y, [1, -1])
maxx = np.max(x)
minx = np.min(x)
maxy = np.max(y)
miny = np.min(y)
if problem in {"IDP", "INP"}:
    xx = np.linspace(minx, maxx, 500)
    yy = np.linspace(miny, maxy, 500)
    pointx, pointy = np.meshgrid(xx, yy)
    index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.03, I=True)
else:
    xx = np.linspace(minx - (maxx - minx) / 2, maxx + (maxx - minx) / 2, 500)
    yy = np.linspace(miny - (maxy - miny) / 2, maxy + (maxy - miny) / 2, 500)
    pointx, pointy = np.meshgrid(xx, yy)
    index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.03, I=False)
out_data = np.concatenate([x1, y1], axis=1)

L = getattr(tn, problem)(M, para)
# _, u_true = IDP.f_to_pde(f_fourier, out_data)
phi_true_f = np.reshape(phi_true_f, [1, -1])
# f_f = rt.resort_fourier(np.reshape(f_fourier / np.sqrt(2 * np.pi), [1, -1]), N)
input = (np.concatenate([x, y], axis=1), np.reshape(f, [1, -1]), np.reshape(t, [-1, 1]))
phi_predict = np.reshape(model._outputs([], input), [1, -1])
phi_predict_fourier = np.reshape(
    np.fft.fft(phi_predict * np.sqrt(2 * np.pi) / M), [1, -1]
)
phi_predict_f = tn.resort_fourier(phi_predict_fourier, N)


plt.figure(figsize=(8, 6))
plt.plot(
    np.reshape(t1, [-1, 1]),
    np.reshape(tn.to_point(phi_true_f, t1), [-1, 1]),
    linewidth=5.0,
    label="true",
)
plt.plot(
    np.reshape(t, [-1, 1]),
    np.reshape(phi_predict, [-1, 1]),
    linewidth=5.0,
    label="predict",
    linestyle="dashed",
)
plt.legend()
plt.xlabel(r"$t$", font={"size": 25})
plt.ylabel(r"$\varphi$(t)", font={"size": 25})
plt.title("The output of BI-DeepONet", font={"size": 25})
plt.tight_layout(pad=0.2)
# plt.savefig(
#     f"../Figures/{name}_phi_1.eps",
#     format="eps",
# )
# plt.savefig(
#     f"../Figures/{name}_phi_1.pdf",
#     format="pdf",
# )

plt.figure(figsize=(8, 6))
plt.plot(np.reshape(tn.to_point(f_f, t1), [-1, 1]), linewidth=5.0)
plt.title(r"$\widetilde{f}(t)$", font={"size": 25})
plt.xlabel(r"$t$", font={"size": 25})
plt.ylabel(r"$\widetilde{f}(t)$", font={"size": 25})

plt.tight_layout(pad=0.2)


plt.figure(figsize=(8, 6))
plt.plot(
    np.reshape(tn.to_point(px, t1), [-1, 1]),
    np.reshape(tn.to_point(py, t1), [-1, 1]),
    linewidth=5.0,
)
plt.title("boundary", font={"size": 25})
plt.tight_layout(pad=0.2)

plt.figure(figsize=(8, 6))
plt.plot(
    np.reshape(phi_predict, [-1, 1]) - np.reshape(tn.to_point(phi_true_f, t), [-1, 1]),
    label="Error",
)
plt.title("Error of BI-DeepONet", font={"size": 25})
plt.tight_layout(pad=0.2)

# phi_true = np.reshape(phi_true, [-1, 1])
print(
    "Example 1: MAE of phi is ===>",
    np.linalg.norm((phi_predict_f) - (phi_true_f)),
)
print(
    "Example 1: MRE of phi is ===>",
    np.linalg.norm((phi_predict_f) - (phi_true_f)) / np.linalg.norm((phi_true_f)),
)
phi_predict = np.reshape(phi_predict, [1, -1])
phi_true = np.reshape(phi_true_f, [1, -1])
u_predict = L.phi_to_pde(phi_predict_f, out_data)
u_true = L.phi_to_pde(phi_true_f, out_data)
mse = np.linalg.norm((u_predict - u_true))
rse = np.linalg.norm((u_predict - u_true)) / np.linalg.norm((u_true))
print("Example 1: MAE of u is ===>", mse)
print("Example 1: MRE of u is ===>", rse)
x = np.reshape(x, [1, -1])
y = np.reshape(y, [1, -1])
U_true = tn.block(index, u_true)
U_pred = tn.block(index, u_predict)
X = pointx
Y = pointy

plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, U_true, cmap="jet", shading="gouraud")
# plt.contourf(X,Y,Z_true)
# plt.colorbar(label="AUPR")
plt.title("True", font={"size": 25})
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout(pad=0.2)


plt.figure(figsize=(8, 6))

plt.pcolormesh(
    X,
    Y,
    U_pred,
    cmap="jet",
    shading="gouraud",
)
# plt.contourf(X,Y,Z_true)
plt.title("Predict", font={"size": 25})
plt.xlabel(r"$x$", font={"size": 25})
plt.ylabel(r"$y$", font={"size": 25})
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout(pad=0.2)
# plt.savefig(
#     f"../Figures/{name}_pred_1.pdf",
#     format="pdf",
# )

plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    abs(U_true - U_pred),
    cmap="jet",
    shading="gouraud",
    # vmin=-0.02,
    # vmax=0.06,
)
# plt.contourf(X,Y,Z_true)
plt.title("Error", font={"size": 25})
plt.xlabel(r"$x$", font={"size": 25})
plt.ylabel(r"$y$", font={"size": 25})
# plt.colorbar(label="AUPR")
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout(pad=0.2)
# plt.savefig(
#     f"../Figures/{name}_error_1.pdf",
#     format="pdf",
# )

para = np.reshape(para_test[r2], [1, -1])
phi_true_f = np.reshape(phi_test[r2], [1, -1])
f_f = np.reshape(f_test[r2], [1, -1])
px = np.reshape(para[0, : 2 * N + 1], [1, -1])
py = np.reshape(para[0, 2 * N + 1 :], [1, -1])
x = tn.to_point(px, t)
y = tn.to_point(py, t)
f = tn.to_point(f_f, t)
f_fourier = np.fft.fft(f) * np.sqrt(2 * np.pi) / M
x = np.reshape(x, [1, -1])
y = np.reshape(y, [1, -1])
maxx = np.max(x)
minx = np.min(x)
maxy = np.max(y)
miny = np.min(y)
if problem in {"IDP", "INP"}:
    xx = np.linspace(minx, maxx, 500)
    yy = np.linspace(miny, maxy, 500)
    pointx, pointy = np.meshgrid(xx, yy)
    index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.03, I=True)
else:
    xx = np.linspace(minx - (maxx - minx) / 2, maxx + (maxx - minx) / 2, 500)
    yy = np.linspace(miny - (maxy - miny) / 2, maxy + (maxy - miny) / 2, 500)
    pointx, pointy = np.meshgrid(xx, yy)
    index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.03, I=False)
out_data = np.concatenate([x1, y1], axis=1)

L = getattr(tn, problem)(M, para)
# _, u_true = IDP.f_to_pde(f_fourier, out_data)
phi_true_f = np.reshape(phi_true_f, [1, -1])
input = (
    np.concatenate([x, y], axis=1),
    np.reshape(f, [1, -1]),
    np.reshape(trunk, [-1, 1]),
)
phi_predict = np.reshape(model._outputs([], input), [1, -1])
phi_predict_fourier = np.reshape(
    np.fft.fft(phi_predict * np.sqrt(2 * np.pi) / M), [1, -1]
)
phi_predict_f = tn.resort_fourier(phi_predict_fourier, N)


plt.figure(figsize=(8, 6))

plt.plot(
    np.reshape(t1, [-1, 1]),
    np.reshape(tn.to_point(phi_true_f, t1), [-1, 1]),
    linewidth=5.0,
    label="true",
)
plt.plot(
    np.reshape(t, [-1, 1]),
    np.reshape(phi_predict, [-1, 1]),
    linewidth=5.0,
    label="predict",
    linestyle="dashed",
)
plt.legend()
plt.xlabel(r"$t$", font={"size": 25})
plt.ylabel(r"$\varphi$(t)", font={"size": 25})
plt.title("The output of BI-DeepONet", font={"size": 25})
plt.tight_layout(pad=0.2)
# plt.savefig(
#     f"../Figures/{name}_phi_2.eps",
#     format="eps",
# )
# plt.savefig(
#     f"../Figures/{name}_phi_2.pdf",
#     format="pdf",
# )

plt.figure(figsize=(8, 6))
plt.plot(np.reshape(tn.to_point(f_f, t1), [-1, 1]), linewidth=5.0)
plt.title(r"$\widetilde{f}(t)$", font={"size": 25})
plt.xlabel(r"$t$", font={"size": 25})
plt.ylabel(r"$\widetilde{f}(t)$")
plt.tight_layout(pad=0.2)

plt.figure(figsize=(8, 6))
plt.plot(
    np.reshape(tn.to_point(px, t1), [-1, 1]),
    np.reshape(tn.to_point(py, t1), [-1, 1]),
    linewidth=5.0,
)
plt.title("boundary", font={"size": 25})
plt.tight_layout(pad=0.2)

plt.figure(figsize=(8, 6))
plt.plot(
    np.reshape(phi_predict, [-1, 1]) - np.reshape(tn.to_point(phi_true_f, t), [-1, 1]),
    label="Error",
)
plt.title("Error of BI-DeepONet", font={"size": 25})
plt.tight_layout(pad=0.2)

# phi_true = np.reshape(phi_true, [-1, 1])
print(
    "Example 2: MAE of phi is ===>",
    np.linalg.norm((phi_predict_f) - (phi_true_f)),
)
print(
    "Example 2: MRE of phi is ===>",
    np.linalg.norm((phi_predict_f) - (phi_true_f)) / np.linalg.norm((phi_true_f)),
)
phi_predict = np.reshape(phi_predict, [1, -1])
phi_true = np.reshape(phi_true_f, [1, -1])
u_predict = L.phi_to_pde(phi_predict_f, out_data)
u_true = L.phi_to_pde(phi_true_f, out_data)
mse = np.linalg.norm((u_predict - u_true))
rse = np.linalg.norm((u_predict - u_true)) / np.linalg.norm((u_true))
print("Example 2: MAE of u is ===>", mse)
print("Example 2: MRE of u is ===>", rse)
x = np.reshape(x, [1, -1])
y = np.reshape(y, [1, -1])
U_true = tn.block(index, u_true)
U_pred = tn.block(index, u_predict)
X = pointx
Y = pointy


plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, U_true, cmap="jet", shading="gouraud")
# plt.contourf(X,Y,Z_true)
# plt.colorbar(label="AUPR")
plt.title("True", font={"size": 25})
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout(pad=0.2)


plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    U_pred,
    cmap="jet",
    shading="gouraud",
)
# plt.contourf(X,Y,Z_true)
plt.title("Predict", font={"size": 25})
plt.xlabel(r"$x$", font={"size": 25})
plt.ylabel(r"$y$", font={"size": 25})
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
# plt.colorbar(label="AUPR")
plt.tight_layout(pad=0.2)
# plt.savefig(
#     f"../Figures/{name}_pred_2.pdf",
#     format="pdf",
# )

plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    abs(U_true - U_pred),
    cmap="jet",
    shading="gouraud",
    # vmin=-0.02,
    # vmax=0.06,
)
# plt.contourf(X,Y,Z_true)
plt.title("Error", font={"size": 25})
plt.xlabel(r"$x$", font={"size": 25})
plt.ylabel(r"$y$", font={"size": 25})
# plt.colorbar(label="AUPR")
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout(pad=0.2)
# plt.savefig(
#     f"../Figures/{name}_error_2.pdf",
#     format="pdf",
# )
plt.show()
