"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle"""

import random
import deepxde as dde
import numpy as np
import test_Numerical as tn
import time
import scipy.io as sio
import os
import matplotlib.pyplot as plt

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=fusible"
begin = time.time()
path = "/home/ext8/mengbin/BI-TDONet_datasets/LBVP/data_exterior_Dirichlet_problem"
data = sio.loadmat(path + "/data.mat")
para = data["para"]
phi = data["phi"]
f = data["f"]
m = f.shape[0]
random.seed(1024)
idx = np.array(range(m))
random.shuffle(idx)
para = para[idx, :]
phi = phi[idx, :]
f = f[idx, :]
end = time.time()
print("准备数据用时：", end - begin)
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

trunk = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
# trunk=np.linspace(0,2*N+1)
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
train_f = tn.to_point(f_train, trunk)
test_f = tn.to_point(f_test, trunk)

# density function
train_phi = tn.to_point(phi_train, trunk)
test_phi = tn.to_point(phi_test, trunk)

# show data shape
trunk = np.reshape(trunk, [-1, 1])

print(para_train.shape, f_train.shape, trunk.shape, phi_train.shape)

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
net = dde.nn.MIONetCartesianProd(
    [2 * M, 300, 300, 300, 300],
    [M, 300, 300, 300, 300],
    [dim_x, 300, 300, 300, 300],
    "relu",
    "Glorot normal",
)

model = dde.Model(data, net)
name = "BI-DeepONet_EDP"
# Initialize network
iterations = 1500
batch_size = 8192

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
plt.rcParams.update(
    {
        # "text.usetex": True,
        # "text.latex.preamble": r"\usepackage{bm}",
        "font.size": 30,
        "font.weight": "bold",
    }
)
dde.utils.plot_loss_history(losshistory)
plt.tight_layout()  # 使用 tight_layout 自动调整

# loss_path = "LOSS/DeepONet_EDP/%s" % name
# dde.utils.save_loss_history(losshistory, loss_path)

################################################################
# testing
################################################################
m1 = f_test.shape[0]

MRE = []
MAE = []
terror = []
NN = 10
Nn = m1 // NN
# true value
# test_phi = tn.to_point(phi_test,trunk)
print(test_phi.shape[1], phi_test.shape[1])
for i in range(NN):
    a = Nn * i
    b = Nn * (i + 1)
    input = (test_para[a:b, :], test_f[a:b, :], trunk)
    begin = time.time()
    phi_predict = model._outputs([], input)
    end = time.time()
    terror.append(end - begin)

    phi_predict_fourier = np.fft.fft(phi_predict) * np.sqrt(2 * np.pi) / M
    phi_predict_f = tn.resort_fourier(phi_predict_fourier, N)
    error = phi_test[a:b, :] - phi_predict_f
    MAE.append((np.linalg.norm(error, axis=1)))
    MRE.append(np.linalg.norm(error, axis=1) / np.linalg.norm(phi_test[a:b, :], axis=1))

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
f_f = f_test[r : r + 1, :]
phi_true_f = phi_test[r : r + 1, :]


# f = 2 * f
f = tn.to_point(f_f, t)
f_fourier = np.fft.fft(f) * np.sqrt(2 * np.pi) / M
x = np.reshape(x, [1, -1])
y = np.reshape(y, [1, -1])
maxx = np.max(x)
minx = np.min(x)
maxy = np.max(y)
miny = np.min(y)
xx = np.linspace(minx - (maxx - minx) / 2, maxx + (maxx - minx) / 2, 500)
yy = np.linspace(miny - (maxy - miny) / 2, maxy + (maxy - miny) / 2, 500)
pointx, pointy = np.meshgrid(xx, yy)

index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.03, I=False)
out_data = np.concatenate([x1, y1], axis=1)

input = (np.concatenate([x, y], axis=1), np.reshape(f, [1, -1]), np.reshape(t, [-1, 1]))
phi_predict = np.reshape(model._outputs([], input), [1, -1])

phi_predict_fourier = np.reshape(
    np.fft.fft(phi_predict * np.sqrt(2 * np.pi) / M), [1, -1]
)
phi_predict_f = tn.resort_fourier(phi_predict_fourier, N)
EDP = tn.EDP(M, N, para_f)
u_predict = EDP.phi_to_pde(phi_predict_f, out_data)
U_pred = tn.block(index, u_predict)
u_true = EDP.phi_to_pde(phi_true_f, out_data)
U_true = tn.block(index, u_true)

X = pointx
Y = pointy


plt.figure(figsize=(8, 6))
plt.plot(
    np.reshape(t1, [-1, 1]),
    np.reshape(tn.to_point(phi_true_f, t1), [-1, 1]),
    linewidth=5.0,
    label="true",
)
plt.plot(
    np.reshape(t1, [-1, 1]),
    np.reshape(tn.to_point(phi_predict_f, t1), [-1, 1]),
    linewidth=5.0,
    label="predict",
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
    np.reshape(tn.to_point(f_f, t1), [-1, 1]),
    linewidth=5.0,
)
plt.title(r"$\widetilde{f}(t)$")
plt.xlabel(r"$t$")
plt.ylabel(r"$\widetilde{f}(t)$")
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


# plt.figure(figsize=(8, 6))
# plt.plot(
#     np.reshape(tn.to_point(phi_predict_f, t1), [-1, 1])
#     - np.reshape(tn.to_point(phi_true_f, t1), [-1, 1]),
#     label="Error",
# )
# plt.title("Error of BI-DeepONet")
# plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整

print(
    "Example: MNE of phi is ===>",
    np.linalg.norm((phi_predict_f) - (phi_true_f)),
)
print(
    "Example: MRE of phi is ===>",
    np.linalg.norm((phi_predict_f) - (phi_true_f)) / np.linalg.norm((phi_true_f)),
)


plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    U_true,
    cmap="jet",
    shading="gouraud",
)  # 彩虹热力图
plt.title("True")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout()  # 使用 tight_layout 自动调整

plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    U_pred,
    cmap="jet",
    shading="gouraud",
)  # 彩虹热力图
plt.title("Predict")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
cb = plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout()  # 使用 tight_layout 自动调整

plt.figure(figsize=(8, 6))
plt.pcolormesh(
    X,
    Y,
    abs(U_true - U_pred),
    cmap="jet",
    shading="gouraud",
)  # 彩虹热力图
plt.title("Error")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.colorbar()
# cb.ax.tick_params(labelsize=20)
plt.tight_layout()  # 使用 tight_layout 自动调整
plt.show()
