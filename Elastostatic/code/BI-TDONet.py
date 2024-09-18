import random
import time
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from keras import Model
import test_Numerical as tn
import os
import preFDONet as pfno
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # 设置TensorFlow只分配必需的显存空间
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)
# /home/ext8/mengbin
begin = time.time()
path = "/home/ext8/mengbin/BI-TDONet_datasets/Elastostatic_problem"
data = sio.loadmat(path + "/data.mat")
para = data["para"]
f = data["f"]
u = data["u"]
# E=data["E"]
# nu=data["nu"]
du1 = data["du1"]
du2 = data["du2"]
M = 128
m = f.shape[0]
N = (f.shape[1] - 2) // 4
random.seed(1024)
idx = np.array(range(m))
para = para[idx, :]
u = u[idx, :]
f = f[idx, :]
du1 = du1[idx, :]
du2 = du2[idx, :]
end = time.time()
print("准备数据用时：", end - begin)
para_train = para[0 : 8 * m // 10, :]
para_test = para[8 * m // 10 :, :]
u_train = u[0 : 8 * m // 10, :]
u_test = u[8 * m // 10 :, :]
f_train = f[0 : 8 * m // 10, :]
f_test = f[8 * m // 10 :, :]
du1_train = du1[0 : 8 * m // 10, :]
du1_test = du1[8 * m // 10 :, :]
du2_train = du2[0 : 8 * m // 10, :]
du2_test = du2[8 * m // 10 :, :]

X_train = np.concatenate([para_train, f_train], axis=1)
X_test = np.concatenate([para_test, f_test], axis=1)
Y_train = u_train
Y_test = u_test

print(X_train.shape, Y_train.shape)
if __name__ == "__main__":
    name = "BI-TDONet_elastostatic"
    Layers = [
        [8 * N + 4, 600, 600, 600, 600, 4 * N + 2],
        [4 * N + 2, 600, 600, 600, 600, 4 * N + 2],
        [4 * N + 2 + 4 * N + 2, 600, 600, 600, 600, 4 * N + 2],
    ]
    # N_train = 5000
    batch_size = 3200
    epochs = 5
    learning_rate = 0.002
    start = time.time()
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = pfno.model(Layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # "adam",
        loss=
        # "mse"
        model.rse_coefficient_mean,
    )

    checkpoint_save_path = "model/%s/%s.ckpt" % (name, name)
    if os.path.exists(checkpoint_save_path + ".index"):
        print("-------------load the model-----------------")
        model.load_weights(checkpoint_save_path)

    # 设置保存模型参数
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        monitor="loss",
        mode="min",
        save_weights_only=True,
        save_best_only=True,
    )

    # # 设置学习率衰减策略，patience和min_delta参考sciann。
    # patience = max(20, epochs // 100)
    patience = 20
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=patience, min_delta=0.0, min_lr=1e-18
    )

    # 训练，batchsize=64时每个epoch训练时间大概为1s
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, Y_test),
        validation_split=0,
        validation_freq=1,
        callbacks=[cp_callback, lr_callback],
        shuffle=True,
        verbose=2,
    )
    end = time.time()
    print("Model train time: %.2fs" % (end - start))

    # 输出结果
    model.summary()
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.rcParams.update(
        {
            # "text.usetex": True,
            # "text.latex.preamble": r"\usepackage{bm}",
            "font.size": 30,
            "font.weight": "bold",
        }
    )
    plt.figure()
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training  Loss")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    ###################################
    ###################################
    ############ Example ##############
    ###################################
    t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
    t1 = np.linspace(0, 2 * np.pi, M)
    m1 = f_test.shape[0]

    MAE = []
    MRE = []
    terror = []
    NN = 10
    Nn = m1 // NN
    for i in range(NN):
        a = Nn * i
        b = Nn * (i + 1)
        input = np.concatenate([para_test[a:b, :], u_test[a:b, :]], axis=1)
        begin = time.time()
        phi_predict = model.out(input.astype(np.float32))
        end = time.time()
        terror.append(end - begin)
        error = Y_test[a:b, :] - phi_predict
        MAE.append((np.linalg.norm(error, axis=1)))
        MRE.append(
            np.linalg.norm(error, axis=1) / np.linalg.norm(Y_test[a:b, :], axis=1)
        )

        # if m1%NN !=0:
        #     input = np.concatenate([para_test[b:,:],f_test[b:,:]],axis=1)
        #     begin = time.time()
        #     phi_predict = model.out(input.astype(np.float32))
        #     end = time.time()
        #     terror.append(end - begin)
        #     error=Y_test[b:,:]-phi_predict
        #     MSE.append((np.linalg.norm(error, axis=1))**2/Y_test.shape[1])
        #     RSE.append(np.linalg.norm(error, axis=1)/np.linalg.norm(Y_test[b:,:], axis=1))
        mean_MAE = np.mean(MAE)
        mean_MRE = np.mean(MRE)
        var_MAE = np.var(MAE)
        var_MRE = np.var(MRE)

    print("系数平均MSE=", mean_MAE)
    print("系数平均RSE=", mean_MRE)
    print("系数方差MSE=", var_MAE)
    print("系数方差RSE=", var_MRE)
    print("平均推理时间=%s milliseconds" % (np.sum(terror) / m1 * 1000))

    E = 1
    nu = 0.3
    r = random.randint(0, m1)
    para = para_test[r : r + 1, :]
    u_true_f = np.reshape(u_test[r], [1, -1])
    f_f = np.reshape(f_test[r], [1, -1])
    du1 = np.reshape(du1_test[r], [1, -1])
    du2 = np.reshape(du2_test[r], [1, -1])
    px = np.reshape(para[0, : 2 * N + 1], [1, -1])
    py = np.reshape(para[0, 2 * N + 1 :], [1, -1])
    x, y, dx, dy, _, _ = tn.initial(para, t)
    f1, f2, _, _, _, _ = tn.initial(f_f, t)
    u1, u2, _, _, _, _ = tn.initial(u_true_f, t)

    # x = tn.to_point(px, t)
    # y = tn.to_point(py, t)
    # f = tn.to_point(f_f, t)
    # f_fourier = np.fft.fft(f) * np.sqrt(2 * np.pi) / M
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
    ELA = tn.Elastostatic(para, E, nu, N)

    input = np.concatenate([para, u_true_f], axis=1)
    f_predict = np.reshape(model.out(input.astype(np.float32)), [1, -1])

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(u_true_f[:, : 2 * N + 1], t1), [-1, 1]),
        label="u1",
        linewidth=5.0,
    )
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(u_true_f[:, 2 * N + 1 :], t1), [-1, 1]),
        label="u2",
        linewidth=5.0,
    )
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.1)
    plt.title("Displacement", fontweight="bold")
    # plt.xlabel(r"$\theta$", fontweight="bold")
    # plt.ylabel(r"$u(\gamma(\theta))$")
    plt.legend(loc="best")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(f_f[:, : 2 * N + 1], t1), [-1, 1]),
        linewidth=5.0,
        label="f1 true",
    )
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(f_predict[:, : 2 * N + 1], t1), [-1, 1]),
        label="f1 predict",
        linewidth=5.0,
        linestyle="dashed",
    )
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(f_f[:, 2 * N + 1 :], t1), [-1, 1]),
        linewidth=5.0,
        label="f2 true",
    )
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(f_predict[:, 2 * N + 1 :], t1), [-1, 1]),
        label="f2 predict",
        linewidth=5.0,
        linestyle="dashed",
    )
    plt.legend(loc="best")
    # plt.xlabel(r"$\theta$", fontweight="bold")
    # plt.ylabel(r"$t(\gamma(\theta))$", fontweight="bold")
    plt.title("The output of BI-TDONet", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    # phi_true = np.reshape(phi_true, [-1, 1])
    print(
        "Example: MAE of phi is ===>",
        np.linalg.norm((f_predict) - (f_f)),
    )
    print(
        "Example: MRE of phi is ===>",
        np.linalg.norm((phi_predict) - (f_f)) / np.linalg.norm((f_f)),
    )
    f_predict = np.reshape(f_predict, [1, -1])
    f_true = np.reshape(f_f, [1, -1])
    u_predict1, u_predict2 = ELA.to_pde(u_true_f, f_predict, x1.T, y1.T)

    # u_true1, u_true2 = ELA.to_pde(u_true_f, f_f, x1.T, y1.T)
    u_true1 = du1[:, 0] * x1 + du1[:, 1] * y1
    u_true2 = du2[:, 0] * x1 + du2[:, 1] * y1
    u_true1x = du1[:, 0] * x + du1[:, 1] * y
    u_true2x = du2[:, 0] * x + du2[:, 1] * y
    u_true1 = u_true1.T
    u_true2 = u_true2.T
    u_predict = np.concatenate([u_predict1, u_predict2], axis=0)
    u_true = np.concatenate([u_true1, u_true2], axis=1)
    mae = np.linalg.norm((u_predict - u_true))
    mre = np.linalg.norm((u_predict - u_true)) / np.linalg.norm((u_true))
    print("Example: MAE of u is ===>", mae)
    print("Example: MRE of u is ===>", mre)

    U_true1 = tn.block(index, u_true1)
    U_pred1 = tn.block(index, u_predict1)
    U_true2 = tn.block(index, u_true1)
    U_pred2 = tn.block(index, u_predict2)

    X = pointx
    Y = pointy

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, U_true1, cmap="jet", shading="gouraud")  # 彩虹热力图
    plt.title(r"$u_1$ True", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        U_pred1,
        cmap="jet",
        shading="gouraud",
    )  # 彩虹热力图
    plt.title(r"$u_1$ Predict", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        abs(U_true1 - U_pred1),
        cmap="jet",
        shading="gouraud",
        # vmin=-0.02,
        # vmax=0.06,
    )  # 彩虹热力图
    # plt.xlabel(r"$x$", fontweight="bold")
    # plt.ylabel(r"$y$", fontweight="bold")
    plt.title(r"$u_1$ Error", fontweight="bold")
    plt.colorbar()
    # plt.axis("equal")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, U_true2, cmap="jet", shading="gouraud")  # 彩虹热力图
    plt.title(r"$u_2$ True", fontweight="bold")
    # plt.colorbar()
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        U_pred2,
        cmap="jet",
        shading="gouraud",
    )  # 彩虹热力图
    # plt.xlabel(r"$x$", fontweight="bold")
    # plt.ylabel(r"$y$", fontweight="bold")
    plt.title(r"$u_2$ Predict", fontweight="bold")
    # plt.colorbar()
    # plt.axis("equal")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        abs(U_true2 - U_pred2),
        cmap="jet",
        shading="gouraud",
        # vmin=-0.02,
        # vmax=0.06,
    )  # 彩虹热力图
    # plt.xlabel(r"$x$", fontweight="bold")
    # plt.ylabel(r"$y$", fontweight="bold")
    plt.title(r"$u_2$ Error", fontweight="bold")
    plt.colorbar()
    # plt.axis("equal")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U_true1, U_true2)
    plt.title(r"$u$ True", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U_pred1, U_pred2)
    plt.title(r"$u_2$ Predict", fontweight="bold")
    # plt.colorbar()
    # plt.axis("equal")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        np.sqrt((U_true1 - U_pred1) ** 2 + (U_true2 - U_pred2) ** 2),
        cmap="jet",
        shading="gouraud",
        # vmin=-0.02,
        # vmax=0.06,
    )  # 彩虹热力图
    # plt.xlabel(r"$x$", fontweight="bold")
    # plt.ylabel(r"$y$", fontweight="bold")
    plt.title(r"$u_2$ Error", fontweight="bold")
    plt.colorbar()
    # plt.axis("equal")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整
    plt.show()
