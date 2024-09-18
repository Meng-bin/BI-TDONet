"""
@author: Bin Meng

"""

import random
import time
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import test_Numerical as tn
import os
import preFDONet as pfon
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # 设置TensorFlow只分配必需的显存空间
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 打印异常
        print(e)

begin = time.time()
path = "/home/ext8/mengbin/BI-TDONet_datasets/LBVP/data_exterior_Dirichlet_problem"
data = sio.loadmat(path + "/data.mat")
para = data["para"]
phi = data["phi"]
f = data["f"]

M = 128
m = f.shape[0]
N = (f.shape[1] - 1) // 2
random.seed(1024)
idx = np.array(range(m))
random.shuffle(idx)
para = para[idx, :]
phi = phi[idx, :]
f = f[idx, :]
end = time.time()
print("准备数据用时：", end - begin)
print(para.shape, phi.shape, f.shape)
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

sio.savemat(
    path + "/data_test.mat",
    {"para": para_test, "phi": phi_test, "f": f_test},
)
print(X_train.shape, Y_train.shape)
if __name__ == "__main__":
    name = "BI-TDONet_EDP"
    Layers = [
        [6 * N + 3, 300, 300, 300, 300, 2 * N + 1],
        [4 * N + 2, 300, 300, 300, 300, 2 * N + 1],
        [2 * N + 1 + 4 * N + 2, 300, 300, 300, 300, 2 * N + 1],
    ]
    # N_train = 5000
    batch_size = 8192
    epochs = 5
    learning_rate = 0.001
    weight_decay = 0.01
    start = time.time()
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = pfon.model(Layers)
    print("Layers=", Layers)
    print(
        "epochs= %d, batchs=%d, lr= %3f,weight_decay=%3f"
        % (epochs, batch_size, learning_rate, weight_decay)
    )
    print(name)
    # cosine_decay = tf.keras.experimental.CosineDecay(
    #             initial_learning_rate=0.001, decay_steps=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        # "adam",
        loss=model.rse_coefficient_mean,
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

    # 设置学习率衰减策略，patience和min_delta参考sciann。
    patience = max(20, epochs // 100)
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=patience, min_delta=0.0, min_lr=1e-18
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
    # sio.savemat("LOSS/BI-TDONet_EDP/%s.mat"%name, {"loss": loss, "val_loss": val_loss})
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
        input = np.concatenate([para_test[a:b, :], f_test[a:b, :]], axis=1)
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

    r = random.randint(0, m1)
    para = np.reshape(para_test[r], [1, -1])
    phi_true_f = np.reshape(phi_test[r], [1, -1])
    f_f = np.reshape(f_test[r], [1, -1])
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
    # xx = np.linspace(minx, maxx, 500)
    # yy = np.linspace(miny, maxy, 500)
    xx = np.linspace(minx - (maxx - minx) / 2, maxx + (maxx - minx) / 2, 500)
    yy = np.linspace(miny - (maxy - miny) / 2, maxy + (maxy - miny) / 2, 500)
    pointx, pointy = np.meshgrid(xx, yy)
    index, x1, y1 = tn.determine(x, y, pointx, pointy, min=0.03, I=False)
    out_data = np.concatenate([x1, y1], axis=1)
    EDP = tn.EDP(M, N, para)
    _, u_true = EDP.f_to_pde(f_fourier, out_data)
    phi_true_f = np.reshape(phi_true_f, [1, -1])
    # f_f = rt.resort_fourier(np.reshape(f_fourier / np.sqrt(2 * np.pi), [1, -1]), N)
    input = np.concatenate([para, f_f], axis=1)
    start = time.time()
    phi_predict = np.reshape(model.out(input.astype(np.float32)), [1, -1])
    input1 = np.concatenate([2 * para, 2 * f_f], axis=1)
    start = time.time()
    phi_predict1 = np.reshape(model.out(input1.astype(np.float32)), [1, -1])
    end = time.time()
    plt.figure()
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(phi_predict, t1), [-1, 1]),
        label="predict",
    )
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(phi_true_f, t1), [-1, 1]),
        label="true",
    )
    plt.legend()
    plt.title("The output of BI-FDONet")
    plt.figure()
    plt.plot(np.reshape(tn.to_point(f_f, t1) / -2, [-1, 1]))
    plt.title("\widetilde{f}")
    plt.figure()
    plt.plot(
        np.reshape(tn.to_point(px, t1), [-1, 1]),
        np.reshape(tn.to_point(py, t1), [-1, 1]),
    )
    plt.title("boundary")
    plt.figure()
    plt.plot(
        np.reshape(tn.to_point(phi_predict, t1), [-1, 1])
        - np.reshape(tn.to_point(phi_true_f, t1), [-1, 1]),
        label="Error",
    )
    plt.title("Error of BI-FDONet")
    print("Model Time is:", (end * 1000 - start * 1000))
    # phi_true = np.reshape(phi_true, [-1, 1])
    print(
        "Example: MAE of phi is ===>",
        np.linalg.norm((phi_predict) - (phi_true_f)) ** 2 / phi_true_f.shape[0],
    )
    print(
        "Example: MRE of phi is ===>",
        np.linalg.norm((phi_predict) - (phi_true_f)) / np.linalg.norm((phi_true_f)),
    )
    phi_predict = np.reshape(phi_predict, [1, -1])
    phi_true = np.reshape(phi_true_f, [1, -1])
    u_predict = EDP.phi_to_pde(phi_predict, out_data)
    mae = np.linalg.norm((u_predict - u_true))
    mre = np.linalg.norm((u_predict - u_true)) / np.linalg.norm((u_true))
    print("Example: MAE of u is ===>", mae)
    print("Example: MRE of u is ===>", mre)
    x = np.reshape(x, [1, -1])
    y = np.reshape(y, [1, -1])
    U_true = tn.block(index, u_true)
    U_pred = tn.block(index, u_predict)
    X = pointx
    Y = pointy
    plt.figure()
    plt.pcolormesh(X, Y, U_true, cmap="jet", shading="gouraud")  # 彩虹热力图
    # plt.contourf(X,Y,Z_true)
    plt.colorbar(label="AUPR")
    plt.title("True")
    plt.figure()
    plt.pcolormesh(
        X,
        Y,
        U_pred,
        cmap="jet",
        shading="gouraud",
    )  # 彩虹热力图
    # plt.contourf(X,Y,Z_true)
    plt.title("Predict")
    plt.colorbar(label="AUPR")
    plt.figure()
    plt.pcolormesh(
        X,
        Y,
        U_true - U_pred,
        cmap="jet",
        shading="gouraud",
        # vmin=-0.02,
        # vmax=0.06,
    )  # 彩虹热力图
    # plt.contourf(X,Y,Z_true)
    plt.title("Error")
    plt.colorbar(label="AUPR")
    plt.show()
