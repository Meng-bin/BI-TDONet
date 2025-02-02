import os
import time
import numpy as np
import tqdm
import test_Numerical as tn
import pandas as pd


def mb(M, N, weight_decay, datanumber, cur=False, seed=10):
    """

    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)"
    weight_decay: Variance decay rate of coefficient. Namely,
        variance of coefficient whith cosx and sinx is 1,
        variance of coefficient whith cos2x and sin2x is weight_decay,
        variance of coefficient whith cosNx and sinNx is weight_decay^N.
    data_number: number of boundaries
    path: Data Storage Location
    cur: limit the max curvature less than cur
    O: Do you want to delete the data already stored in the path? True is Yes, False is no.
    """
    np.random.seed(seed)
    print("Generating boundaries ... ...")
    begin = time.time()
    t = np.linspace(0, 2 * np.pi, M, endpoint=False)
    sin_part = []
    cos_part = []
    dsin_part = []
    dcos_part = []
    ddsin_part = []
    ddcos_part = []
    for i in range(1, N + 1):
        sin_part.append(np.sin(i * t))
        cos_part.append(np.cos(i * t))
        dsin_part.append(i * np.cos(i * t))
        dcos_part.append(-i * np.sin(i * t))
        ddsin_part.append(-(i**2) * np.sin(i * t))
        ddcos_part.append(-(i**2) * np.cos(i * t))

    data_number = 1000000
    if data_number < datanumber * 10:
        data_number = 100 * datanumber
    # data_number = datanumber

    Px = np.random.randn(data_number, 1)
    Py = np.random.randn(data_number, 1)
    for j in range(N):
        nextx = np.reshape(
            np.random.normal(0, weight_decay**j, (data_number, 2)),
            [data_number, -1],
        )
        nexty = np.reshape(
            np.random.normal(0, weight_decay**j, (data_number, 2)),
            [data_number, -1],
        )
        Px = np.concatenate([Px, nextx], axis=1)
        Py = np.concatenate([Py, nexty], axis=1)
    idx = np.array(range(data_number))
    np.random.shuffle(idx)
    Px = Px[idx, :]
    Py = Py[idx, :]
    Px_cos = Px[:, 1::2]
    Px_sin = Px[:, 2::2]
    Py_cos = Py[:, 1::2]
    Py_sin = Py[:, 2::2]
    P = np.concatenate(
        [
            np.reshape(Px[:, 0], [-1, 1]),
            Px_cos,
            Px_sin,
            np.reshape(Py[:, 0], [-1, 1]),
            Py_cos,
            Py_sin,
        ],
        axis=1,
    )
    x, y, dx, dy, ddx, ddy = tn.initial(P, t)
    curvature = (dx * ddy - ddx * dy) / ((dx**2 + dy**2) ** (3 / 2))
    index = 0
    data = []
    for i in range(x.shape[0]):
        xx = np.reshape(x[i, :], [1, -1])
        yy = np.reshape(y[i, :], [1, -1])
        if tn.judge(xx, yy) == False and (
            cur == False or np.max(np.abs(curvature[i, :])) < cur
        ):
            index = index + 1
            p = np.reshape(P[i, :], [1, -1])
            # data = pd.DataFrame(p)
            data.append(p)
            if np.mod(index, 100) == 0:
                end = time.time()
                print(
                    "Step now: %d, %d boundaries are stored. Using time  %5f s"
                    % (i, index, end - begin)
                )
            # data.to_csv(
            #     path + "/" + name,
            #     sep=" ",
            #     index=0,
            #     header=0,
            #     mode="a",
            # )
            if index == datanumber:
                break
    end = time.time()
    print("Total time is : %5f s" % (end - begin))
    print("Total data is : %d " % (index))
    return data
