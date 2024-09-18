"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import random
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import test_Numerical as tn
import time
import scipy.io as sio
import os
import tensorflow as tf


"""
In elastostatic problems, the displacement u on the boundary is
used as the input of the network, and the stress f on the boundary
is used as the output of the network.
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# tf.config.list_physical_devices('GPU')
begin = time.time()
path = "/home/ext8/mengbin/BI-TDONet_datasets/Elastostatic_problem"
data = sio.loadmat(path + "/data.mat")
para = data["para"]
u = data["u"]
f = data["f"]
du1 = data["du1"]
du2 = data["du2"]
m = f.shape[0]
random.seed(1024)
idx = np.array(range(m))
random.shuffle(idx)
para = para[idx, :]
u = u[idx, :]
f = f[idx, :]
end = time.time()
print("准备数据用时：", end - begin)
M = 128
m = f.shape[0]
N = (f.shape[1] - 1) // 2

print(para.shape, u.shape, f.shape)

para_train = para[0 : 8 * m // 10, :]
para_test = para[8 * m // 10 :, :]
u_train = u[0 : 8 * m // 10, :]
u_test = u[8 * m // 10 :, :]
f_train = f[0 : 8 * m // 10, :]
f_test = f[8 * m // 10 :, :]
du1_test = du1[8 * m // 10 :, :]
du2_test = du2[8 * m // 10 :, :]
# initialization
M = 128
N = 20

trunk = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)

# branch data 1 of boundary with Fourier coefficient
train_para_x = para_train[:, : 2 * N + 1]
train_para_y = para_train[:, 2 * N + 1 :]
test_para_x = para_test[:, : 2 * N + 1]
test_para_y = para_test[:, 2 * N + 1 :]

train_para = np.concatenate(
    [tn.to_point(train_para_x, trunk), tn.to_point(train_para_y, trunk)], axis=1
)
test_para = np.concatenate(
    [tn.to_point(test_para_x, trunk), tn.to_point(test_para_y, trunk)], axis=1
)

# branch data 2 of Dirchlet boundary condition with f
train_f1, train_f2, _, _, _, _ = tn.initial(f_train, trunk)
test_f1, test_f2, _, _, _, _ = tn.initial(f_test, trunk)

train_f = np.concatenate([train_f1, train_f2], axis=1)
test_f = np.concatenate([test_f1, test_f2], axis=1)
# density function
train_u1, train_u2, _, _, _, _ = tn.initial(u_train, trunk)
test_u1, test_u2, _, _, _, _ = tn.initial(u_test, trunk)

train_u = np.concatenate([train_u1, train_u2], axis=1)
test_u = np.concatenate([test_u1, test_u2], axis=1)
# show data shape
trunk = np.reshape(trunk, [-1, 1])
# trunk=np.concatenate([trunk,trunk],axis=0)

# packed data
# print(np.concatenate([trunk,trunk],axis=0))

X_train = (train_para, train_u, trunk)
y_train = train_f1
X_test = (test_para, test_u, trunk)
y_test = test_f1

data = dde.data.QuadrupleCartesianProd(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)

dim_x = 1
Layers = [
    [2 * M, 600, 600, 600, 600],
    [2 * M, 600, 600, 600, 600],
    [dim_x, 600, 600, 600, 600],
]
# set net structure

net = dde.nn.MIONetCartesianProd(
    Layers[0],
    Layers[1],
    Layers[2],
    "relu",
    "Glorot normal",
)

model = dde.Model(data, net)
name = "BI-DeepONet_elastostatic_f1"
# Initialize network
iterations = 1500
batch_size = 3200
print(name, iterations, batch_size)

# Compile and Train
model.compile(
    "adam",
    lr=0.001,
    decay=("inverse time", iterations // 100, 0.5),
    metrics=["mean l2 relative error"],
)
checkpoint_save_path = "model/%s/%s-%s.ckpt" % (name, name, iterations)
if os.path.exists(checkpoint_save_path + ".index"):
    print("-------------load the model-----------------")
    model.restore(checkpoint_save_path, device=None, verbose=0)
losshistory, train_state = model.train(iterations=iterations, batch_size=batch_size)
model.save("model/%s/%s" % (name, name), protocol="backend", verbose=0)
# Visualization

dde.utils.plot_loss_history(losshistory)
plt.tight_layout()  # 使用 tight_layout 自动调整

# loss_path = "LOSS/DeepONet_EDP/%s" % name
# dde.utils.save_loss_history(losshistory, loss_path)

################################################################
# testing
################################################################
m1 = f_test.shape[0]
E = 1
nu = 0.3
MRE = []
MAE = []
terror = []
NN = 10
Nn = m1 // NN
# true value
# test_phi = tn.to_point(u_test,trunk)
for i in range(NN):
    a = Nn * i
    b = Nn * (i + 1)
    input = (test_para[a:b, :], test_u[a:b, :], trunk)
    begin = time.time()
    f_predict = model._outputs([], input)
    end = time.time()
    terror.append(end - begin)

    f_predict_fourier = np.fft.fft(f_predict) * np.sqrt(2 * np.pi) / M
    f_predict_f = tn.resort_fourier(f_predict_fourier, N)
    error = f_test[a:b, : 2 * N + 1] - f_predict_f
    MAE.append((np.linalg.norm(error, axis=1)))
    MRE.append(np.linalg.norm(error, axis=1) / np.linalg.norm(f_test[a:b, :], axis=1))

mean_MRE = np.mean(MRE)
mean_MAE = np.mean(MAE)
var_MRE = np.var(MRE)
var_MAE = np.var(MAE)

print("系数平均MRE=", mean_MRE)
print("系数平均RSE=", mean_MAE)
print("系数方差MRE=", var_MRE)
print("系数方差RSE=", var_MAE)
print("平均推理时间=%s ms" % (np.sum(terror) * 1000 / m1))
# print(terror)

t = trunk
t1 = np.linspace(0, 2 * np.pi, M)

r = random.randint(0, m1)
para_f = para_test[r : r + 1, :]
px = np.reshape(para_f[:, : 2 * N + 1], [1, -1])
py = np.reshape(para_f[:, 2 * N + 1 :], [1, -1])
x = tn.to_point(px, t)
y = tn.to_point(py, t)
f_true_f = f_test[r : r + 1, :]
u_f = u_test[r : r + 1, :]


# f = 2 * f
u1, u2, _, _, _, _ = tn.initial(u_f, t)
u = np.concatenate([u1, u2], axis=1)
# f_true_fourier = np.fft.fft(f) * np.sqrt(2 * np.pi) / M
x = np.reshape(x, [1, -1])
y = np.reshape(y, [1, -1])
maxx = np.max(x)
minx = np.min(x)
maxy = np.max(y)
miny = np.min(y)
xx = np.linspace(minx, maxx, 500)
yy = np.linspace(miny, maxy, 500)
# xx = np.linspace(minx - (maxx - minx) / 2, maxx + (maxx - minx) / 2, 500)
# yy = np.linspace(miny - (maxy - miny) / 2, maxy + (maxy - miny) / 2, 500)
pointx, pointy = np.meshgrid(xx, yy)

index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.03, I=True)
out_data = np.concatenate([x1, y1], axis=1)

input = (np.concatenate([x, y], axis=1), np.reshape(u, [1, -1]), np.reshape(t, [-1, 1]))
f_predict = np.reshape(model._outputs([], input), [1, -1])

f_predict_fourier = np.reshape(np.fft.fft(f_predict * np.sqrt(2 * np.pi) / M), [1, -1])
f_predict_f1 = tn.resort_fourier(f_predict_fourier, N)

##### ************** #####
# f_pred,f_predict_f2 needs to be stored, read in and concatenated to form f_predict_f
# To ensure the normal operation of the program, we temporarily use f_predict_f1 instead of f_predict_f2

f_predict_f = np.concatenate([f_predict_f1, f_predict_f1], axis=1)
ELA = tn.Elastostatic(para_f, E, nu, N)
u_predict1, u_predict2 = ELA.to_pde(u_f, f_predict_f, x1.T, y1.T)
U_pred1 = tn.block(index, u_predict1)
U_pred2 = tn.block(index, u_predict2)
u_true1 = du1[r, 0] * x1 + du1[r, 1] * y1
u_true2 = du2[r, 0] * x1 + du2[r, 1] * y1
U_true1 = tn.block(index, u_true1)
U_true2 = tn.block(index, u_true2)
X = pointx
Y = pointy

plt.rcParams.update(
    {
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{bm}",
        "font.size": 30,
        "font.weight": "bold",
    }
)

plt.figure(figsize=(8, 6))
plt.plot(
    np.reshape(t1, [-1, 1]),
    np.reshape(tn.to_point(f_true_f[:, : 2 * N + 1], t1), [-1, 1]),
    linewidth=5.0,
    label="t1 true",
)
plt.plot(
    np.reshape(t, [-1, 1]),
    np.reshape(f_predict, [-1, 1]),  ####f_predict1
    label="t1 predict",
    linewidth=5.0,
    linestyle="dashed",
)
plt.plot(
    np.reshape(t1, [-1, 1]),
    np.reshape(tn.to_point(f_true_f[:, 2 * N + 1 :], t1), [-1, 1]),
    linewidth=5.0,
    label="t2 true",
)
plt.plot(
    np.reshape(t, [-1, 1]),
    np.reshape(f_predict, [-1, 1]),  ####f_predict2
    label="t2 predict",
    linewidth=5.0,
    linestyle="dashed",
)
plt.legend()
plt.xlabel(r"$t$")
plt.ylabel(r"$\varphi$(t)")
plt.title("The output of BI-DeepONet")
plt.tight_layout()  # 使用 tight_layout 自动调整


plt.figure(figsize=(8, 6))
plt.plot(
    np.reshape(t1, [-1, 1]),
    np.reshape(tn.to_point(u_f[:, : 2 * N + 1], t1), [-1, 1]),
    label="u1",
    linewidth=5.0,
)
plt.plot(
    np.reshape(t1, [-1, 1]),
    np.reshape(tn.to_point(u_f[:, 2 * N + 1 :], t1), [-1, 1]),
    label="u2",
    linewidth=5.0,
)
plt.title("Displacement", fontweight="bold")
plt.xlabel(r"$\theta$", fontweight="bold")
plt.ylabel(r"$u(\gamma(\theta))$")
plt.legend(loc="best")
plt.tight_layout()  # 使用 tight_layout 自动调整

plt.figure(figsize=(8, 6))
plt.plot(
    np.reshape(tn.to_point(px, t1), [-1, 1]),
    np.reshape(tn.to_point(py, t1), [-1, 1]),
    linewidth=5.0,
)
plt.xlabel(r"$\widetilde{{\gamma}}_1(t)$")
plt.ylabel(r"$\widetilde{{\gamma}}_2(t)$")
plt.title("boundary")
plt.tight_layout()  # 使用 tight_layout 自动调整


## This paragraph needs to be modified accordingly

# f1_fourier = tn.resort_fourier(f1_predict_fourier, N)
# f2_fourier = tn.resort_fourier(f2_predict_fourier, N)
# f_predict_f = np.concatenate([f1_fourier, f2_fourier], axis=1)
# f_predict = np.reshape(f_predict_f, [1, -1])
# print(
#     "Example: MNE of phi is ===>",
#     np.linalg.norm((f_predict_f) - (f_true_f)),
# )
# print(
#     "Example: MRE of phi is ===>",
#     np.linalg.norm((f_predict_f) - (f_true_f)) / np.linalg.norm((u_f)),
# )


plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    U_true1,
    cmap="jet",
    shading="gouraud",
)  # 彩虹热力图
plt.title("True1")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout()  # 使用 tight_layout 自动调整

plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    U_true2,
    cmap="jet",
    shading="gouraud",
)  # 彩虹热力图
plt.title("True2")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout()  # 使用 tight_layout 自动调整

plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    U_pred1,
    cmap="jet",
    shading="gouraud",
)  # 彩虹热力图
plt.title("Predict1")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout()  # 使用 tight_layout 自动调整

plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    U_pred2,
    cmap="jet",
    shading="gouraud",
)  # 彩虹热力图
plt.title("Predict2")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout()  # 使用 tight_layout 自动调整

plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    abs(U_true1 - U_pred1),
    cmap="jet",
    shading="gouraud",
)  # 彩虹热力图
plt.title("Error1")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout()  # 使用 tight_layout 自动调整

plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    abs(U_true2 - U_pred2),
    cmap="jet",
    shading="gouraud",
)  # 彩虹热力图
plt.title("Error2")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout()  # 使用 tight_layout 自动调整

plt.show()
