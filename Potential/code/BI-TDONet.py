"""
Learning operators using BI-TDONet

Additional details: https://doi.org/10.48550/arXiv.2406.02298
"""

import random
import time
import h5py
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os
import scipy.io as sio
import sys
from keras import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

current_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_path, "../../src/")
if src_path in sys.path:
    sys.path.remove(src_path)
sys.path.insert(0, src_path)
from utilities3 import *
import test_Numerical as tn


class TDONet(Model):
    def __init__(self, Layers):
        super(TDONet, self).__init__()

        self.Layers_branch1 = Layers[0]
        self.Layers_branch2 = Layers[1]
        self.Layers_inner = Layers[2]
        # Initialize NN
        (
            self.Weights_branch1,
            self.Biases_branch1,
            self.Weights_branch2,
            self.Biases_branch2,
            self.Weight_out,
            self.Biases_out,
        ) = self.initialize_NN(Layers)

    def initialize_NN(self, Layers):
        Layers1 = Layers[0]
        Layers2 = Layers[1]
        Layers3 = Layers[2]
        weights_branch1 = []
        biases_branch1 = []
        weights_branch2 = []
        biases_branch2 = []
        weights_out = []
        biases_out = []

        for l in range(0, len(Layers1) - 1):
            W = self.xavier_init(size=[Layers1[l], Layers1[l + 1]])
            b = tf.Variable(
                tf.zeros([1, Layers1[l + 1]], dtype=tf.float32), dtype=tf.float32
            )
            weights_branch1.append(W)
            biases_branch1.append(b)

        for l in range(0, len(Layers2) - 1):
            W = self.xavier_init(size=[Layers2[l], Layers2[l + 1]])
            b = tf.Variable(
                tf.zeros([1, Layers2[l + 1]], dtype=tf.float32), dtype=tf.float32
            )
            weights_branch2.append(W)
            biases_branch2.append(b)

        for l in range(0, len(Layers3) - 1):
            W = self.xavier_init(size=[Layers3[l], Layers3[l + 1]])
            b = tf.Variable(
                tf.zeros([1, Layers3[l + 1]], dtype=tf.float32), dtype=tf.float32
            )
            weights_out.append(W)
            biases_out.append(b)

        return (
            weights_branch1,
            biases_branch1,
            weights_branch2,
            biases_branch2,
            weights_out,
            biases_out,
        )

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal(
                [in_dim, out_dim],
                stddev=xavier_stddev,
                dtype=tf.float32,
            ),
        )

    def he_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        he_stddev = np.sqrt(2 / in_dim)  # He initialization standard deviation
        return tf.Variable(
            tf.random.truncated_normal([in_dim, out_dim], stddev=he_stddev),
            dtype=tf.float32,
        )

    @tf.function(jit_compile=True)
    def oper_net(self, X):

        weights_branch1 = self.Weights_branch1
        biases_branch1 = self.Biases_branch1
        weights_branch2 = self.Weights_branch2
        biases_branch2 = self.Biases_branch2
        weights_out = self.Weight_out
        biases_out = self.Biases_out

        num_Layers1 = len(weights_branch1)
        num_Layers2 = len(weights_branch2)
        num_Layers3 = len(weights_out)

        mid = []
        # branch1=gamma
        branch1 = X[:, : self.Layers_branch2[0]]
        branch2 = X[:, self.Layers_branch2[0] :]

        mid.append(X)
        for l in range(0, num_Layers1 - 1):
            W = weights_branch1[l]
            b = biases_branch1[l]
            mid.append(tf.nn.relu(tf.add(tf.matmul(mid[l], W), b)))
            # mid.append(tf.nn.tanh(tf.add(tf.matmul(mid[l], W), b)))
            # mid.append(tf.nn.sigmoid(tf.add(tf.matmul(mid[l], W), b)))
        W = weights_branch1[-1]
        b = biases_branch1[-1]
        Y_branch1 = tf.add(tf.matmul(mid[-1], W), b)

        mid = []
        mid.append(branch1)
        for l in range(0, num_Layers2 - 1):
            W = weights_branch2[l]
            b = biases_branch2[l]
            mid.append(tf.nn.relu(tf.add(tf.matmul(mid[l], W), b)))
            # mid.append(tf.nn.tanh(tf.add(tf.matmul(mid[l], W), b)))
            # mid.append(tf.nn.sigmoid(tf.add(tf.matmul(mid[l], W), b)))

        W = weights_branch2[-1]
        b = biases_branch2[-1]
        Y_branch2 = tf.add(tf.matmul(mid[-1], W), b)
        out = Y_branch1 * Y_branch2
        out = tf.concat([branch1, out], axis=1)

        mid = []
        mid.append(out)
        for l in range(0, num_Layers3 - 1):
            W = weights_out[l]
            b = biases_out[l]
            mid.append(tf.nn.relu(tf.add(tf.matmul(mid[l], W), b)))
            # mid.append(tf.nn.tanh(tf.add(tf.matmul(mid[l], W), b)))
            # mid.append(tf.nn.sigmoid(tf.add(tf.matmul(mid[l], W), b)))
        # W = weights_out[-2]
        # b = biases_out[-2]
        # mid.append(tf.sin(tf.add(tf.matmul(mid[-1], W), b)))
        W = weights_out[-1]
        b = biases_out[-1]
        out = tf.add(tf.matmul(mid[-1], W), b)
        # + branch2
        return out

    def rse_coefficients_mean(self, y_true, y_pred):
        return tf.reduce_mean(
            tf.norm(y_pred - y_true, axis=1) / tf.norm(y_true, axis=1)
        )

    def call(self, X):
        output = self.oper_net(X)
        return output

    # @tf.function(jit_compile=True)
    def out(self, X):
        out = self.oper_net(X)
        return out.numpy()


gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Set TensorFlow to allocate only the necessary video memory space
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Printing exception
        print(e)
begin = time.time()

path = "/home/ext8/mengbin/BI-TDONet_update/potential"
data_path = path + "/data_potential.mat"
print("Data location:", data_path)
data1 = sio.loadmat(data_path)

path = "/home/ext8/mengbin/BI-TDONet_update/LBVP/data_exterior_Neumann_problem"
data_path = path + "/data.mat"
print("Data location:", data_path)
data2 = sio.loadmat(data_path)

para = np.concatenate([data1["para"], data2["para"]], axis=0)
phi = np.concatenate([data1["phi"], data2["phi"]], axis=0)
f = np.concatenate([data1["f"], data2["f"]], axis=0)


problem = "ENP"

M = 128
m = f.shape[0]
N = (f.shape[1] - 1) // 2
np.random.seed(1117)
idx = np.array(range(m))
np.random.shuffle(idx)
para = para[idx, :]
phi = phi[idx, :]
f = f[idx, :]

# para = cp.asnumpy(para)
# phi = cp.asnumpy(phi)
# f = cp.asnumpy(f)

print("Original data types:")
print("para dtype:", para.dtype)
print("f dtype:", f.dtype)
print("phi dtype:", phi.dtype)

print(
    f"Shape of boundary dataset: {para.shape}\n",
    f"Shape of density dataset: {phi.shape}\n",
    f"Shape of rhs dataset:{f.shape}",
)

para_train = para[0 : 8 * m // 10, :]
para_test = para[8 * m // 10 :, :]
phi_train = phi[0 : 8 * m // 10, :]
phi_test = phi[8 * m // 10 :, :]
f_train = f[0 : 8 * m // 10, :]
f_test = f[8 * m // 10 :, :]


X_train = np.concatenate([para_train, f_train], axis=1)
X_test = np.concatenate([para_test, f_test], axis=1)

Y_train = phi_train
Y_test = phi_test


print("dataset types:")
print("Xtrain dtype:", X_train.dtype)
print("X_test dtype:", X_test.dtype)
print("Y_train dtype:", Y_train.dtype)
print("Y_test dtype:", Y_test.dtype)

print(
    "Shape of Training Dataset :",
    X_train.shape,
    Y_train.shape,
)

print(
    "Shape of Testing Dataset :",
    X_test.shape,
    Y_test.shape,
)

if __name__ == "__main__":
    name = "potential_BI-TDONet_0113"
    Layers = [
        [6 * N + 3, 600, 600, 600, 600],
        [4 * N + 2, 600, 600, 600, 600],
        [600 + 4 * N + 2, 600, 600, 600, 600, (2 * N + 1)],
    ]
    batch_size = 2**13
    epochs = 5000
    learning_rate = 0.001
    lr_decay = 0.5
    # state = "train"
    state = "test"

    start = time.time()
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = TDONet(Layers)
    print("Layers=", Layers)
    print(
        "epochs= %d, batchs=%d, lr= %3f,lr_decay=%3f"
        % (epochs, batch_size, learning_rate, lr_decay)
    )
    print(name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # "adam",
        loss=model.rse_coefficients_mean,
        # loss=model.myloss,
        # loss="mse",
    )

    checkpoint_save_path = "model/%s/%s.ckpt" % (name, name)
    if os.path.exists(checkpoint_save_path + ".index"):
        print("-------------load the model-----------------")
        model.load_weights(checkpoint_save_path)

    # Set and save model parameters
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        monitor="loss",
        mode="min",
        save_weights_only=True,
        save_best_only=True,
    )

    # Set learning rate decay strategy.
    patience = max(20, epochs // 100)
    # patience = 50
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=lr_decay, patience=patience, min_delta=0.0, min_lr=1e-18
    )

    if state == "train":
        start = time.time()
        # # Training
        history = model.fit(
            X_train,
            Y_train,
            # train_dataset,
            batch_size=batch_size,
            # dataset,
            epochs=epochs,
            validation_data=(X_test, Y_test),
            # validation_data=test_dataset,
            validation_split=0,
            validation_freq=1,
            # callbacks=cp_callback,
            callbacks=[cp_callback, lr_callback],
            shuffle=True,
            verbose=2,
        )
        end = time.time()
        print("Model train time: %.2fs" % (end - start))

        # Output result
        model.summary()
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        sio.savemat(f"loss/{name}.mat", {"loss": loss, "val_loss": val_loss})
        model_loss = sio.loadmat(f"loss/{name}.mat")
    elif state == "test":
        model_loss = sio.loadmat(f"loss/{name}.mat")
    loss = model_loss["loss"]
    val_loss = model_loss["val_loss"]
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
        np.reshape(np.log10(loss) / 1, [-1, 1]),
        linewidth=5.0,
        label="BI-TDONet \n Training loss",
    )
    plt.plot(
        np.reshape(np.log10(val_loss) / 1, [-1, 1]),
        linewidth=5.0,
        label="BI-TDONet \n Testing loss",
    )
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("lg(MRE)", fontweight="bold")
    plt.legend()
    plt.title("BI-TDONet Loss", fontweight="bold")
    plt.tight_layout(pad=0)
    # plt.savefig(
    #     f"../Figures/{name}_loss.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    # )
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
    NN = 10
    Nn = m1 // NN
    MAE = []
    MRE = []
    terror = []
    input = X_test[1:2, :]
    phi_predict = model.out(input.astype(np.float32))
    # Main loop for full groups
    for i in range(NN):
        a = Nn * i
        b = Nn * (i + 1) if i < NN - 1 else m1  # Adjust b for the last group
        # input = np.concatenate([para_test[a:b, :], f_test[a:b, :]], axis=1)
        input = X_test[a:b, :]
        begin = time.time()
        phi_predict = model.out(input.astype(np.float32))
        end = time.time()
        terror.append(end - begin)
        error = Y_test[a:b, :] - phi_predict

        MAE.append(np.linalg.norm(error, axis=1))
        MRE.append(
            np.linalg.norm(error, axis=1) / np.linalg.norm(Y_test[a:b, :], axis=1)
        )

    terror = np.hstack(terror)  # Stack the results along a new axis (vertically)
    MAE = np.hstack(MAE)
    MRE = np.hstack(MRE)
    # Calculate mean and variance
    mean_MAE = np.mean(MAE)
    mean_MRE = np.mean(MRE)
    var_MAE = np.var(MAE)
    var_MRE = np.var(MRE)

    print("Average MAE = %.4e" % mean_MAE)
    print("Average MRE = %.4e" % mean_MRE)
    print("Variance of MAE = %.4e" % var_MAE)
    print("Variance of MRE = %.4e" % var_MRE)
    print("Average inference time = %.4e milliseconds" % (np.sum(terror) / m1 * 1000))

    ###################################
    ############ Examples ##############
    ###################################
    random.seed(1218)
    v0 = 3
    print("m1 is ", m1)
    r1 = random.randint(0, m1)
    print("r1 is ", r1)
    parameter = np.reshape(para_test[r1], [1, -1])
    x, y, dx, dy, _, _ = tn.initial(parameter, t)
    n1, n2 = dy, -dx
    if np.sum(x * dy - y * dx) < 0:
        n1, n2 = -dy, -n2
    f_point = -2 * v0 * n1 / np.sqrt(n1**2 + n2**2)
    f_fourier = np.fft.fft(f_point) * np.sqrt(2 * np.pi) / M
    f_f = tn.resort_fourier(f_fourier, N)
    input = np.concatenate([parameter, f_f], axis=1)
    x = np.reshape(x, [1, -1])
    y = np.reshape(y, [1, -1])
    maxx = np.max(x)
    minx = np.min(x)
    maxy = np.max(y)
    miny = np.min(y)

    xx = np.linspace(minx - 4 * (maxx - minx) / 2, maxx + 4 * (maxx - minx) / 2, 500)
    yy = np.linspace(miny - 4 * (maxy - miny) / 2, maxy + 4 * (maxy - miny) / 2, 500)
    pointx, pointy = np.meshgrid(xx, yy)
    index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.003, I=False)

    out_data = np.concatenate([x1, y1], axis=1)
    L = getattr(tn, problem)(M, parameter)
    phi_true_f = -L.f_to_pde(f_fourier).reshape(1, -1)
    phi_predict_f = -np.reshape(model.out(input.astype(np.float32)), [1, -1])
    A = np.sum(tn.to_point(phi_predict_f, t) * np.sqrt(dx**2 + dy**2) / M) / np.sum(
        np.sqrt(dx**2 + dy**2) / M
    )
    phi_predict_f[0, 0] = phi_predict_f[0, 0] - A

    plt.figure()
    plt.plot(
        np.array(range(2 * N + 1)).reshape(-1, 1),
        (phi_true_f).reshape(-1, 1),
        label="true",
    )
    plt.plot(
        np.array(range(2 * N + 1)).reshape(-1, 1),
        (phi_predict_f).reshape(-1, 1),
        label="pred",
    )
    plt.tight_layout(pad=0)
    plt.legend()
    plt.title("phi coefficients")

    plt.figure()
    plt.plot(
        np.array(range(2 * N + 1)).reshape(-1, 1),
        (phi_true_f - phi_predict_f).reshape(-1, 1),
    )
    plt.tight_layout(pad=0)

    plt.figure()
    plt.plot(
        np.array(range(2 * N + 1)).reshape(-1, 1),
        np.abs(phi_true_f - phi_predict_f).reshape(-1, 1),
    )
    plt.tight_layout(pad=0)

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
    # plt.xlabel(r"$t$", font={"size": 40})
    # plt.ylabel(r"$\varphi$(t)", font={"size": 40})
    plt.title("BI-TDONet output  ", font={"size": 40})
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_phi_1.eps",
        format="eps",
        bbox_inches="tight",
    )
    # plt.savefig(f"../Figures/{name}_phi_1.pdf", format="pdf", transparent=True)

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(f_f, t1), [-1, 1]),
        linewidth=5.0,
    )
    plt.title(r"$\widetilde{f}(t)$", font={"size": 40})
    # plt.xlabel(r"$t$", font={"size": 40})
    # plt.ylabel(r"$\widetilde{f}(t)$", font={"size": 40})
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_f_1.eps",
        format="eps",
        bbox_inches="tight",
    )
    # plt.savefig(f"../Figures/{name}_f_1.pdf", format="pdf", transparent=True)

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(tn.to_point(phi_predict_f, t1), [-1, 1])
        - np.reshape(tn.to_point(phi_true_f, t1), [-1, 1]),
        label="Error",
    )
    plt.tight_layout(pad=0)
    plt.title("Error of BI-TDONet", font={"size": 40})

    print(
        "Example 1: MAE of phi is ===>",
        np.linalg.norm((phi_predict_f) - (phi_true_f)),
    )
    print(
        "Example 1: MRE of phi is ===>",
        np.linalg.norm((phi_predict_f) - (phi_true_f)) / np.linalg.norm((phi_true_f)),
    )

    phi_predict_f = np.reshape(phi_predict_f, [1, -1])
    phi_true_f = np.reshape(phi_true_f, [1, -1])

    u_true = L.phi_to_pde(phi_true_f, out_data)
    u_predict = L.phi_to_pde(phi_predict_f, out_data)
    mae = np.linalg.norm((u_predict - u_true))
    mre = np.linalg.norm((u_predict - u_true)) / np.linalg.norm((u_true))
    print("Example 1: MAE of u is ===>", mae)
    print("Example 1: MRE of u is ===>", mre)

    x = np.reshape(x, [1, -1])
    y = np.reshape(y, [1, -1])
    U_true = tn.block(index, u_true)
    U_pred = tn.block(index, u_predict)

    X = pointx
    Y = pointy

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, U_true, cmap="jet", shading="gouraud")
    plt.gca().set_axis_off()
    plt.colorbar()
    # plt.xlabel(r"$x$", font={"size": 40})
    # plt.ylabel(r"$y$", font={"size": 40})
    plt.title("True", font={"size": 40})
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_true_1.png",
        format="png",
        bbox_inches="tight",
        transparent=True,
    )

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        U_pred,
        cmap="jet",
        shading="gouraud",
    )
    # plt.xlabel(r"$x$", font={"size": 40})
    # plt.ylabel(r"$y$", font={"size": 40})
    plt.title("BI-TDONet Predict", font={"size": 40})
    plt.gca().set_axis_off()
    plt.colorbar()
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_pred_1.png",
        format="png",
        bbox_inches="tight",
        transparent=True,
    )

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
    # plt.xlabel(r"$x$", font={"size": 40})
    # plt.ylabel(r"$y$", font={"size": 40})
    plt.title("BI-TDONet Error", font={"size": 40})
    plt.gca().set_axis_off()
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_error_1.png",
        format="png",
        bbox_inches="tight",
        transparent=True,
    )

    Vx, Vy = np.gradient(v0 * X + U_true, axis=(1, 0))
    # Vx += v0
    Vx1, Vx2 = np.gradient(v0 * X + U_pred, axis=(1, 0))
    plt.figure(figsize=(8, 6))
    plt.streamplot(X, Y, Vx, Vy, color="black")
    plt.plot(
        np.reshape(x, [-1, 1]),
        np.reshape(y, [-1, 1]),
        linewidth=5.0,
        color="red",
    )
    # 隐藏坐标轴
    plt.gca().set_axis_off()
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.title("True Velocity Field", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整
    plt.savefig(
        f"../Figures/{name}_streamline_true_1.eps",
        format="eps",
        bbox_inches="tight",
        transparent=True,
    )

    plt.figure(figsize=(8, 6))
    plt.streamplot(
        X,
        Y,
        Vx1,
        Vx2,
        integration_direction="both",
        minlength=0.1,
        maxlength=40,
        color="black",
    )
    plt.plot(
        np.reshape(x, [-1, 1]),
        np.reshape(y, [-1, 1]),
        linewidth=5.0,
        color="red",
    )
    # 隐藏坐标轴
    plt.gca().set_axis_off()
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.title("BI-TDONet", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.savefig(
        f"../Figures/{name}_streamline_pred_1.eps",
        format="eps",
        bbox_inches="tight",
        transparent=True,
    )

    r2 = random.randint(0, m1)
    parameter = np.reshape(para_test[r2], [1, -1])
    x, y, dx, dy, _, _ = tn.initial(parameter, t)
    n1, n2 = dy, -dx
    if np.sum(x * dy - y * dx) < 0:
        n1, n2 = -dy, -n2
    f_point = -2 * v0 * n1 / np.sqrt(n1**2 + n2**2)
    f_fourier = np.fft.fft(f_point) * np.sqrt(2 * np.pi) / M
    f_f = tn.resort_fourier(f_fourier, N)
    input = np.concatenate([parameter, f_f], axis=1)

    x = np.reshape(x, [1, -1])
    y = np.reshape(y, [1, -1])
    maxx = np.max(x)
    minx = np.min(x)
    maxy = np.max(y)
    miny = np.min(y)

    xx = np.linspace(minx - 4 * (maxx - minx) / 2, maxx + 4 * (maxx - minx) / 2, 500)
    yy = np.linspace(miny - 4 * (maxy - miny) / 2, maxy + 4 * (maxy - miny) / 2, 500)
    pointx, pointy = np.meshgrid(xx, yy)
    index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.03, I=False)

    out_data = np.concatenate([x1, y1], axis=1)
    L = getattr(tn, problem)(M, parameter)
    phi_true_f = -L.f_to_pde(f_fourier).reshape(1, -1)
    phi_predict_f = -np.reshape(model.out(input.astype(np.float32)), [1, -1])
    A = np.sum(tn.to_point(phi_predict_f, t) * np.sqrt(dx**2 + dy**2) / M) / np.sum(
        np.sqrt(dx**2 + dy**2) / M
    )
    phi_predict_f[0, 0] = phi_predict_f[0, 0] - A
    plt.figure()
    plt.plot(
        np.array(range(2 * N + 1)).reshape(-1, 1),
        (phi_true_f).reshape(-1, 1),
        label="true",
    )
    plt.plot(
        np.array(range(2 * N + 1)).reshape(-1, 1),
        (phi_predict_f).reshape(-1, 1),
        label="pred",
    )
    plt.tight_layout(pad=0)
    plt.legend()
    plt.title("phi coefficients")

    plt.figure()
    plt.plot(
        np.array(range(2 * N + 1)).reshape(-1, 1),
        (phi_true_f - phi_predict_f).reshape(-1, 1),
    )

    plt.figure()
    plt.plot(
        np.array(range(2 * N + 1)).reshape(-1, 1),
        np.abs(phi_true_f - phi_predict_f).reshape(-1, 1),
    )

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
    # plt.xlabel(r"$t$", font={"size": 40})
    # plt.ylabel(r"$\varphi$(t)", font={"size": 40})
    plt.title("BI-TDONet output", font={"size": 40})
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_phi_2.eps",
        format="eps",
        bbox_inches="tight",
    )
    # plt.savefig(f"../Figures/{name}_phi_2.pdf", format="pdf", transparent=True)

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(f_f, t1), [-1, 1]),
        linewidth=5.0,
    )
    plt.title(r"$\widetilde{f}(t)$", font={"size": 40})
    # plt.xlabel(r"$t$", font={"size": 40})
    # plt.ylabel(r"$\widetilde{f}(t)$", font={"size": 40})
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_f_2.eps",
        format="eps",
        bbox_inches="tight",
    )
    # plt.savefig(f"../Figures/{name}_f_2.pdf", format="pdf", transparent=True)

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(tn.to_point(phi_predict_f, t1), [-1, 1])
        - np.reshape(tn.to_point(phi_true_f, t1), [-1, 1]),
        label="Error",
    )
    plt.tight_layout(pad=0)
    plt.title("Error of BI-TDONet", font={"size": 40})

    print(
        "Example 2: MAE of phi is ===>",
        np.linalg.norm((phi_predict_f) - (phi_true_f)),
    )
    print(
        "Example 2: MRE of phi is ===>",
        np.linalg.norm((phi_predict_f) - (phi_true_f)) / np.linalg.norm((phi_true_f)),
    )

    phi_predict_f = np.reshape(phi_predict_f, [1, -1])
    phi_true_f = np.reshape(phi_true_f, [1, -1])
    u_true = L.phi_to_pde(phi_true_f, out_data)
    u_predict = L.phi_to_pde(phi_predict_f, out_data)
    mae = np.linalg.norm((u_predict - u_true))
    mre = np.linalg.norm((u_predict - u_true)) / np.linalg.norm((u_true))
    print("Example 2: MAE of u is ===>", mae)
    print("Example 2: MRE of u is ===>", mre)

    x = np.reshape(x, [1, -1])
    y = np.reshape(y, [1, -1])
    U_true = tn.block(index, u_true)
    U_pred = tn.block(index, u_predict)

    X = pointx
    Y = pointy
    Vx, Vy = np.gradient(v0 * X + U_true, axis=(1, 0))
    Vx1, Vx2 = np.gradient(v0 * X + U_pred, axis=(1, 0))
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, U_true, cmap="jet", shading="gouraud")
    plt.gca().set_axis_off()
    plt.colorbar()
    # plt.xlabel(r"$x$", font={"size": 40})
    # plt.ylabel(r"$y$", font={"size": 40})
    plt.title("True", font={"size": 40})
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_true_2.png",
        format="png",
        bbox_inches="tight",
        transparent=True,
    )

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        U_pred,
        cmap="jet",
        shading="gouraud",
    )
    # plt.xlabel(r"$x$", font={"size": 40})
    # plt.ylabel(r"$y$", font={"size": 40})
    plt.title("BI-TDONet Predict", font={"size": 40})
    plt.gca().set_axis_off()
    plt.colorbar()
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_pred_2.png",
        format="png",
        bbox_inches="tight",
        transparent=True,
    )

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
    # plt.xlabel(r"$x$", font={"size": 40})
    # plt.ylabel(r"$y$", font={"size": 40})
    plt.title("BI-TDONet Error", font={"size": 40})
    plt.gca().set_axis_off()
    cbar = plt.colorbar()
    cbar.formatter.set_powerlimits((0, 0))
    plt.tight_layout(pad=0)
    plt.savefig(
        f"../Figures/{name}_error_2.png",
        format="png",
        bbox_inches="tight",
        transparent=True,
    )

    plt.gca().set_axis_off()
    plt.figure(figsize=(8, 6))
    plt.streamplot(X, Y, Vx, Vy, color="black")
    plt.plot(
        np.reshape(x, [-1, 1]),
        np.reshape(y, [-1, 1]),
        linewidth=5.0,
        color="red",
    )
    # 隐藏坐标轴
    plt.gca().set_axis_off()
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.title("True Velocity Field", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整
    plt.savefig(
        f"../Figures/{name}_streamline_true_2.eps",
        format="eps",
        bbox_inches="tight",
        transparent=True,
    )

    plt.figure(figsize=(8, 6))
    plt.streamplot(
        X,
        Y,
        Vx1,
        Vx2,
        integration_direction="both",
        # minlength=0.1,
        # maxlength=40,
        color="black",
    )
    plt.plot(
        np.reshape(x, [-1, 1]),
        np.reshape(y, [-1, 1]),
        linewidth=5.0,
        color="red",
    )
    # 隐藏坐标轴
    plt.gca().set_axis_off()
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.title("BI-TDONet", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.savefig(
        f"../Figures/{name}_streamline_pred_2.eps",
        format="eps",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()
