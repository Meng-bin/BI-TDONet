import random
import time
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
from keras import Model
from keras.optimizers.schedules import CosineDecay
import test_Numerical as tn
import os
import preFDONet as pfno
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# gpus = tf.config.experimental.list_physical_devices("GPU")
# if gpus:
#     try:
#         # 设置TensorFlow只分配必需的显存空间
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         # 打印异常
#         print(e)
# /home/ext8/mengbin
begin = time.time()
path = "/home/ext8/mengbin/BI-TDONet_datasets/Helmholtz_problem"
data = sio.loadmat(path + "/data_300.mat")
para = data["para"]
f = data["f"]
phi = data["phi"]
k = data["k"]
M = 1024
m = f.shape[0]
Nf = (f.shape[1] - 2) // 4
Np = (para.shape[1] - 2) // 4
end = time.time()
print("准备数据用时：", end - begin)

para_train = para[0 : 8 * m // 10, :]
para_test = para[8 * m // 10 :, :]
phi_train = phi[0 : 8 * m // 10, :]
phi_test = phi[8 * m // 10 :, :]
f_train = f[0 : 8 * m // 10, :]
f_test = f[8 * m // 10 :, :]
k_train = k[0 : 8 * m // 10, :]
k_test = k[8 * m // 10 :, :]

X_train = np.concatenate([para_train, f_train], axis=1)
X_test = np.concatenate([para_test, f_test], axis=1)
Y_train = 10 * phi_train
Y_test = 10 * phi_test

# sio.savemat(
#     path + "/data_test.mat",
#     {"para": para_test, "f": f_test, "phi": phi_test, "k": k_test},
# )
if __name__ == "__main__":
    name = "BI-TDONet_helmholtz"
    Layers = [
        [4 * Np + 2 + 4 * Nf + 2, 2000, 2000],
        [4 * Np + 2, 2000, 2000],
        [4 * Np + 2 + 2000, 2000, 4 * Nf + 2],
    ]
    print(Layers, para.shape, phi.shape)
    # N_train = 5000
    batch_size = 8192
    epochs = 5
    learning_rate = 0.001
    # lr_decayed_fn = CosineDecay(learning_rate, epochs)
    start = time.time()
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = pfno.model(Layers)
    print("Layers=", Layers)
    print("epochs= %d, batchs=%d, lr= %3f" % (epochs, batch_size, learning_rate))
    print(name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            weight_decay=0.01,
            # learning_rate=lr_decayed_fn
        ),
        # "adam",
        loss=
        # "mse"
        model.rse_coefficient_mean,
        # model.myloss
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
    patience = 100
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.1, patience=patience, min_delta=0.0, min_lr=1e-18
    )

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        .shuffle(len(X_train))
        .batch(batch_size)
    )
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(
        batch_size
    )
    # 训练，batchsize=64时每个epoch训练时间大概为1s
    history = model.fit(
        train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=test_dataset,
        validation_split=0,
        validation_freq=1,
        # callbacks=cp_callback,
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
    # true value
    # test_phi = tn.to_point(phi_test,trunk)
    # print(Y_test.shape[1])

    for i in range(NN):
        a = Nn * i
        b = Nn * (i + 1)
        input = np.concatenate([para_test[a:b, :], f_test[a:b, :]], axis=1)
        begin = time.time()
        phi_predict = model.oper_net(input.astype(np.float32)) / 10
        end = time.time()
        terror.append(end - begin)
        error = phi_test[a:b, :] - phi_predict
        MAE.append((np.linalg.norm(error, axis=1)))
        MRE.append(
            np.linalg.norm(error, axis=1) / np.linalg.norm(Y_test[a:b, :], axis=1)
        )

    mean_MAE = np.mean(MAE)
    mean_MRE = np.mean(MRE)
    var_MAE = np.var(MAE)
    var_MRE = np.var(MRE)

    print("系数平均MAE=", mean_MAE)
    print("系数平均MRE=", mean_MRE)
    print("系数方差MAE=", var_MAE)
    print("系数方差MRE=", var_MRE)
    print("平均推理时间=", np.sum(terror) / m1)

    r = random.randint(0, m1)
    para = np.reshape(para_test[r], [1, -1])
    phi_true_f = np.reshape(phi_test[r], [1, -1])
    f_f = np.reshape(f_test[r], [1, -1])
    kk = np.reshape(k_test[r], [1, -1])
    eta = kk
    px = np.reshape(para[0, : 2 * Np + 1], [1, -1])
    py = np.reshape(para[0, 2 * Np + 1 :], [1, -1])
    x = tn.to_point(px, t)
    y = tn.to_point(py, t)
    f = -2 * np.exp(1j * kk * x)
    # f1 = -2 * np.exp(1j * 2 * k * x)
    # f_f = np.fft()
    # f_f = tn.coefficients(f, M, N)
    # f_fourier = np.fft.fft(f) * np.sqrt(2 * np.pi) / M
    x = np.reshape(x, [1, -1])
    y = np.reshape(y, [1, -1])
    maxx = np.max(x)
    minx = np.min(x)
    maxy = np.max(y)
    miny = np.min(y)
    # xx = np.linspace(minx, maxx, 500)
    # yy = np.linspace(miny, maxy, 500)
    xx = np.linspace(minx - 2 * (maxx - minx) / 2, maxx + 2 * (maxx - minx) / 2, 500)
    yy = np.linspace(miny - 2 * (maxy - miny) / 2, maxy + 2 * (maxy - miny) / 2, 500)
    pointx, pointy = np.meshgrid(xx, yy)
    index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.03, I=False)
    out_data = np.concatenate([x1, y1], axis=1)
    IDP = tn.Helmholtz(para, M, kk, eta)
    phi_true_f = np.reshape(phi_true_f, [1, -1])
    phi_real_true, phi_imag_true, _, _, _, _ = tn.initial(phi_true_f, t1)
    phi_true = phi_real_true + 1j * phi_imag_true
    # f_f = rt.resort_fourier(np.reshape(f_fourier / np.sqrt(2 * np.pi), [1, -1]), N)

    input = np.concatenate([para, f_f], axis=1)
    phi_predict_f = np.reshape(model.out(input.astype(np.float32)), [1, -1]) / 10
    phi_real_pred, phi_imag_pred, _, _, _, _ = tn.initial(phi_predict_f, t1)
    phi_predict = phi_real_pred + 1j * phi_imag_pred

    plt.rcParams.update(
        {
            # "text.usetex": True,
            # "text.latex.preamble": r"\usepackage{bm}",
            "font.size": 40,
            "font.weight": "bold",
        }
    )
    plt.rcParams["font.family"] = "serif"

    plt.figure(figsize=(8, 6))
    plt.plot(
        t.T,
        np.real(f).T,
        linewidth=2.0,
    )
    plt.title(r"$\widetilde{f}(t)$: Real part", fontweight="bold")
    # plt.xlabel(r"$t$", fontweight="bold")
    # plt.ylabel(r"$\widetilde{f}(t)$", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.plot(
        t.T,
        np.imag(f).T,
        linewidth=2.0,
    )
    plt.title(r"$\widetilde{f}(t): Imaginary part$", fontweight="bold")
    # plt.xlabel(r"$t$", fontweight="bold")
    # plt.ylabel(r"$\widetilde{f}(t)$", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(phi_real_true, [-1, 1]),
        label="true",
        linewidth=2,
    )
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(phi_real_pred, [-1, 1]),
        linestyle="dashdot",
        label="predict",
        linewidth=2,
    )
    plt.legend(
        loc="best",
        fontsize=15,
    )
    plt.title("BI-TDONet output: Real part", fontweight="bold")
    # plt.xlabel(r"$t$", fontweight="bold")
    # plt.ylabel(r"$\varphi(t)$", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(phi_imag_true, [-1, 1]),
        label="true",
        linewidth=2,
    )
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(phi_imag_pred, [-1, 1]),
        linestyle="dashdot",
        label="predict",
        linewidth=2,
    )
    plt.legend(loc="best", fontsize=15)
    plt.title("BI-TDONet output: Imaginary part", fontweight="bold")
    # plt.xlabel(r"$t$", fontweight="bold")
    # plt.ylabel(r"$\varphi(t)$", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    print(
        "Example : MAE of phi is ===>",
        np.linalg.norm((phi_predict_f) - (phi_true_f)),
    )
    print(
        "Example : MRE of phi is ===>",
        np.linalg.norm((phi_predict_f) - (phi_true_f)) / np.linalg.norm((phi_true_f)),
    )
    phi_predict = np.reshape(phi_predict, [1, -1])
    phi_true = np.reshape(phi_true_f, [1, -1])
    u_predict = IDP.u_scatter(x1.T, y1.T, phi_predict_f)
    u_true = IDP.u_scatter(x1.T, y1.T, phi_true_f)
    mae = np.linalg.norm((u_predict - u_true))
    mre = np.linalg.norm((u_predict - u_true)) / np.linalg.norm((u_true))
    print("Example 1: MSE of u is ===>", mae)
    print("Example 1: RSE of u is ===>", mre)
    u_in = np.exp(1j * kk * x1)
    x = np.reshape(x, [1, -1])
    y = np.reshape(y, [1, -1])
    U_true_real = tn.block(index, np.real(u_true))
    U_true_imag = tn.block(index, np.imag(u_true))
    U_pred_real = tn.block(index, np.real(u_predict))
    U_pred_imag = tn.block(index, np.imag(u_predict))
    U_in_real = tn.block(index, np.real(u_in))
    U_in_imag = tn.block(index, np.imag(u_in))
    U_total_real_true = U_true_real + U_in_real
    U_total_real_pred = U_pred_real + U_in_real
    U_total_imag_true = U_true_imag + U_in_imag
    U_total_imag_pred = U_pred_imag + U_in_imag
    X = pointx
    Y = pointy

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, U_true_real, cmap="jet", shading="gouraud")  # 彩虹热力图
    # plt.colorbar()
    # plt.xlabel(r"$x$", fontweight="bold")
    # plt.ylabel(r"$y$", fontweight="bold")
    plt.title(
        "True: Real part ",
        fontweight="bold",
    )
    plt.gca().set_axis_off()
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, U_true_imag, cmap="jet", shading="gouraud")  # 彩虹热力图
    # plt.colorbar()
    # plt.xlabel(r"$x$", fontweight="bold")
    # plt.ylabel(r"$y$", fontweight="bold")
    plt.title("True: Imaginary part", fontweight="bold")
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        U_pred_real,
        cmap="jet",
        shading="gouraud",
    )  # 彩虹热力图
    plt.title("Predict: Real part", fontweight="bold")
    # plt.xlabel(r"$x$", fontweight="bold")
    # plt.ylabel(r"$y$", fontweight="bold")
    # plt.colorbar()
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        U_pred_imag,
        cmap="jet",
        shading="gouraud",
    )  # 彩虹热力图
    plt.title("Predict: Imaginary part", fontweight="bold")
    # plt.xlabel(r"$x$", fontweight="bold")
    # plt.ylabel(r"$y$", fontweight="bold")
    # plt.colorbar()
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        np.sqrt((U_true_real - U_pred_real) ** 2 + (U_true_imag - U_pred_imag) ** 2),
        cmap="jet",
        shading="gouraud",
        # vmin=-0.02,
        # vmax=0.06,
    )  # 彩虹热力图
    plt.title("Error", fontweight="bold")
    # plt.xlabel(r"$x$", fontweight="bold")
    # plt.ylabel(r"$y$", fontweight="bold")
    plt.colorbar()
    plt.tight_layout(pad=0)  # 使用 tight_layout 自动调整
    plt.show()
    # sio.savemat(
    #     "E:\BI-TDONet\Helmholtz\example_data1.mat",
    #     {
    #         "para": para,
    #         "f": f_f,
    #         "phi_t": phi_true_f,
    #         "u_predict": phi_predict,
    #         "X": X,
    #         "Y": Y,
    #         "Ur_t": U_true_real,
    #         "Ui_t": U_true_imag,
    #         "Ur_p": U_pred_real,
    #         "Ui_p": U_pred_imag,
    #     },
    # )
