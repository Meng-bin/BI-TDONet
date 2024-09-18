"""
Modified based on  Zongyi Li's code

"""

import random
import pandas as pd
from scipy import io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import test_Numerical as tn
import os
import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import scipy


print("\n=============================")
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
print("=============================\n")


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(1, self.width)  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.don = DeepONet(
            [
                [256, 128, 128, 128, 128],
                [128, 128, 128, 128, 128],
                [1, 128, 128, 128, 128],
            ]
        )

    def forward(self, x):

        xd = self.don(x)
        # x = torch.cat([xd.reshape(xd.shape[0],xd.shape[1],1), x[:,:,3:]], dim=2)
        x = xd.reshape(xd.shape[0], xd.shape[1], 1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


################################################################
#  deeponet layer
################################################################
class DeepONet(nn.Module):
    def __init__(self, Layers):
        super(DeepONet, self).__init__()

        self.Layers_branch1 = Layers[0]
        self.Layers_branch2 = Layers[1]
        self.Layers_trunk = Layers[2]

        # Define branch1 network layers
        self.branch_layers1 = nn.ModuleList()
        for i in range(len(self.Layers_branch1) - 1):
            self.branch_layers1.append(
                nn.Linear(self.Layers_branch1[i], self.Layers_branch1[i + 1])
            )

        # Define branch2 network layers
        self.branch_layers2 = nn.ModuleList()
        for i in range(len(self.Layers_branch1) - 1):
            self.branch_layers2.append(
                nn.Linear(self.Layers_branch2[i], self.Layers_branch2[i + 1])
            )

        # Define trunk network layers
        self.trunk_layers = nn.ModuleList()
        for i in range(len(self.Layers_trunk) - 1):
            self.trunk_layers.append(
                nn.Linear(self.Layers_trunk[i], self.Layers_trunk[i + 1])
            )

    def forward(self, x):
        branch_input1_x = x[..., 0]
        branch_input1_y = x[..., 1]
        branch_input1 = torch.cat([branch_input1_x, branch_input1_y], axis=1)
        branch_input2 = x[..., 2]
        trunk_input = x[..., 3]
        trunk_input = trunk_input[0, :].unsqueeze(1)

        # print(branch_input.shape,trunk_input.shape,"I AM TRUNK")

        # Forward pass through the branch network
        for layer in self.branch_layers1[:-1]:
            branch_input1 = torch.relu(layer(branch_input1))
        branch_output1 = self.branch_layers1[-1](branch_input1)

        for layer in self.branch_layers2[:-1]:
            branch_input2 = torch.relu(layer(branch_input2))
        branch_output2 = self.branch_layers2[-1](branch_input2)

        # Forward pass through the trunk network
        for layer in self.trunk_layers[:-1]:
            trunk_input = torch.relu(layer(trunk_input))
        trunk_output = self.trunk_layers[-1](trunk_input)

        branch_output = branch_output1 * branch_output2
        # Combining both outputs
        output = torch.matmul(branch_output, trunk_output.transpose(0, 1))
        return output


class CustomDataset(Dataset):
    def __init__(self, x_train, grid, y_train):
        self.x_train = x_train  # 假设 x_train 是需要批处理的数据部分
        self.grid = grid  # grid 是在每个批次中完整读入的部分
        self.y_train = y_train  # 标签数据

    def __len__(self):
        return len(self.x_train)  # 数据集大小由 x_train 决定

    def __getitem__(self, idx):
        # 返回 x_train 的第 idx 项，完整的 grid，以及 y_train 的第 idx 项
        return (self.x_train[idx], self.grid), self.y_train[idx]


def to_point(p, t):
    if len(p.shape) == 1:
        N = (p.shape[0] - 1) // 2
        p = p.view(1, -1)
    else:
        N = (p.shape[1] - 1) // 2

    sin_part = []
    cos_part = []
    for i in range(1, N + 1):
        sin_part.append(torch.sin(i * t))
        cos_part.append(torch.cos(i * t))

    sin_part = torch.stack(sin_part).view(N, -1)
    cos_part = torch.stack(cos_part).view(N, -1)

    p_cos = p[:, 1 : N + 1].view(-1, N)
    p_sin = p[:, N + 1 : 2 * N + 1].view(-1, N)

    phi = (
        torch.matmul(p_sin, sin_part)
        + torch.matmul(p_cos, cos_part)
        + p[:, 0].view(-1, 1)
    )

    if len(phi.shape) == 1:
        phi = phi.view(1, -1)

    return phi


def rse_loss(output, target):
    # 确保输出和目标具有相同的形状
    loss = torch.sum((output - target) ** 2) / torch.sum(target**2)
    return loss


def FNO_main(train_data_res, save_index):
    """
    Parameters
    ----------
    train_data_res : resolution of the training data
    save_index : index of the saving folder
    """

    ################################################################
    #  configurations
    ################################################################

    s = train_data_res
    # sub = 2**6 #subsampling rate
    sub = 2**13 // s  # subsampling rate (step size)

    batch_size = 11996 // 4
    learning_rate = 0.001

    epochs = 1  # default 500
    step_size = 500  # default 100
    gamma = 0.5

    modes = 10
    width = 128

    ################################################################
    # read training data
    ################################################################

    name = "BI-DeepFNO_EDP"
    print(name)
    # Data is of the shape (number of samples, grid size)
    begin = time.time()
    dataloader = MatReader(
        "/home/ext8/mengbin/BI-TDONet_datasets/LBVP/data_exterior_Dirichlet_problem/data.mat"
    )
    para = dataloader.read_field("para")
    phi = dataloader.read_field("phi")
    f = dataloader.read_field("f")
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
    N = (f.shape[1] - 1) // 2

    para_train = para[0 : 8 * m // 10, :]
    para_test = para[8 * m // 10 :, :]
    phi_train = phi[0 : 8 * m // 10, :]
    phi_test = phi[8 * m // 10 :, :]
    f_train = f[0 : 8 * m // 10, :]
    f_test = f[8 * m // 10 :, :]
    m1 = f_test.shape[0]

    trunk = torch.linspace(0, 2 * np.pi * (1 - 1 / M), M)
    # branch data 1 of boundary with Fourier coefficient
    train_para_x = para_train[:, : 2 * N + 1]
    train_para_y = para_train[:, 2 * N + 1 :]
    test_para_x = para_test[:, : 2 * N + 1]
    test_para_y = para_test[:, 2 * N + 1 :]

    train_para1 = to_point(train_para_x, trunk)
    train_para2 = to_point(train_para_y, trunk)

    test_para1 = to_point(test_para_x, trunk)
    test_para2 = to_point(test_para_y, trunk)

    # branch data 2 of Dirchlet boundary condition with f
    train_f = to_point(f_train, trunk)
    test_f = to_point(f_test, trunk)

    # density function
    train_phi = to_point(phi_train, trunk)
    test_phi = to_point(phi_test, trunk)

    y_train = train_phi
    y_test = test_phi

    grid = torch.reshape(trunk, [-1, 1])
    print("grid_shape", grid.shape)

    ntrain = y_train.shape[0]
    ntest = y_test.shape[0]

    x_train = torch.cat(
        [
            train_para1.reshape(ntrain, s, 1),
            train_para2.reshape(ntrain, s, 1),
            train_f.reshape(ntrain, s, 1),
            grid.repeat(ntrain, 1, 1),
        ],
        dim=2,
    )
    x_test = torch.cat(
        [
            test_para1.reshape(ntest, s, 1),
            test_para2.reshape(ntest, s, 1),
            test_f.reshape(ntest, s, 1),
            grid.repeat(ntest, 1, 1),
        ],
        dim=2,
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    # model
    model = FNO1d(modes, width).cuda()
    checkpoint_save_path = "model/%s" % (name)
    if os.path.exists(checkpoint_save_path + "/%s" % name):
        print("-------------load the model-----------------")
        model.load_state_dict(torch.load(checkpoint_save_path + "/%s" % name))
    total_params = count_params(model)

    ################################################################
    # training and evaluation
    ################################################################
    # lambda_fn = lambda epoch: 1 / (1 + gamma * epoch)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fn)
    # patience = max(20, epochs // 100)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, min_lr=1e-18)

    start_time = time.time()
    myloss = LpLoss(size_average=False)
    train_losses = []
    test_losses = []
    # y_normalizer.cuda()
    # x_train=x_train.cuda()
    # y_train=y_train.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            # x = [item.cuda() for item in x]
            # y=y.cuda()
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(
                out.view(batch_size, -1), y.view(batch_size, -1), reduction="mean"
            )

            # mse.backward()
            # out = y_normalizer.decode(out.view(batch_size, -1))
            # y = y_normalizer.decode(y)
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()
            # l2.backward() # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():

            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                out = model(x)
                # out = y_normalizer.decode(out.view(batch_size, -1))
                test_l2 += myloss(
                    out.view(batch_size, -1), y.view(batch_size, -1)
                ).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        train_losses.append((train_mse, train_l2))
        test_losses.append(test_l2)

        t2 = default_timer()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "Epoch: %d, time: %.3f, Train Loss: %.3e,  Test l2: %.4f, lr: %.4f,"
            % (ep, t2 - t1, train_l2, test_l2, current_lr)
        )
        # print(ep, t2-t1, train_mse, train_l2, test_l2)

    elapsed = time.time() - start_time

    # Save the model state dictionary
    # 创建文件夹（如果不存在）
    os.makedirs(checkpoint_save_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        "/home/mengbin/operator_learning/BI-TDONet/LBVP/model/%s/%s" % (name, name),
    )

    print("\n=============================")
    print("Training done...")
    print("Total number of parameters: ", total_params)
    print("Training time: %.3f" % (elapsed))
    print("=============================\n")

    x_test = x_test.cuda()
    y_test = y_test.cuda()

    # 获取总数据量和每个批次的大小
    total_size = x_test.shape[0]
    batch_size = total_size // 2999

    # 用于存储每个批次的误差
    errors = []
    errors1 = []
    tt = []
    for i in range(2999):

        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        # 获取当前批次的数据
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]

        # 计算模型输出并处理维度
        begin = time.time()
        pred_batch = model(x_batch).cuda()
        end = time.time()
        tt.append(end - begin)
        pred_batch = torch.squeeze(
            pred_batch, axis=2
        )  # 确保预测输出与y_batch的维度一致

        # 计算当前批次的误差
        mse = torch.mean(torch.norm(pred_batch - y_batch, dim=1))
        errors1.append(mse.item())
        batch_error = torch.mean(mse / torch.norm(y_batch, dim=1))
        errors.append(batch_error.item())

    # 计算所有批次的平均误差
    print(
        "The average inference time per sample is %s milliseconds ."
        % (np.sum(tt) * 1000 / total_size)
    )
    average_error1 = sum(errors1) / len(errors1)
    variance1 = np.var(errors1)
    # variance1 = sum((x - average_error1) ** 2 for x in errors1) / len(errors1)
    print(
        "Average l2 relative error and variance across all batches:",
        average_error1,
        variance1,
    )

    average_error = sum(errors) / len(errors)
    # variance = sum((x - average_error) ** 2 for x in errors) / len(errors)
    variance = np.var(errors)
    print(
        "Average l2 Norm error and variance across all batches:",
        average_error,
        variance,
    )
    # ====================================
    # saving settings
    # ====================================
    # current_directory = os.getcwd()
    # resolution = "TrainRes_"+str(train_data_res)
    # folder_index = str(save_index)

    # results_dir = "/results/" + resolution +"/" + folder_index +"/"
    # save_results_to = current_directory + results_dir
    # if not os.path.exists(save_results_to):
    #     os.makedirs(save_results_to)
    # model_dir = "/model/" + resolution +"/" + folder_index +"/"
    # save_models_to = current_directory + model_dir
    # if not os.path.exists(save_models_to):
    #     os.makedirs(save_models_to)

    ################################################################
    # testing
    ################################################################
    t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
    t1 = np.linspace(0, 2 * np.pi, M)
    r = random.randint(0, m1)
    para_f = para_test[r : r + 1, :].numpy()
    px = np.reshape(para_f[:, : 2 * N + 1], [1, -1])
    py = np.reshape(para_f[:, 2 * N + 1 :], [1, -1])
    x = tn.to_point(px, t)
    y = tn.to_point(py, t)
    f_f = f_test[r : r + 1, :].numpy()
    phi_true_f = phi_test[r : r + 1, :].numpy()
    f = tn.to_point(f_f, t)
    # f = 2 * f

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
    # 将 x 和 y 转换为张量，并且调整形状为 (ntrain, s, 1)，统一转换为 float 类型
    x_tensor = torch.tensor(x, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.float)

    # 假设 f 是一个向量，需要重复到每个样本和每个时间步，并转换为 float 类型
    f_tensor = torch.tensor(f, dtype=torch.float)
    # 假设 t 也是一个向量，需要调整形状并转换为 float 类型
    grid = torch.tensor(np.reshape(t, [-1, 1]), dtype=torch.float)

    # 将所有张量在最后一个维度上进行拼接
    input = torch.cat(
        [
            x_tensor.reshape(1, s, 1),
            y_tensor.reshape(1, s, 1),
            f_tensor.reshape(1, s, 1),
            grid.repeat(1, 1, 1),
        ],
        dim=2,
    ).cuda()
    # input = torch.cat([x_tensor, y_tensor, f_tensor, grid], dim=2)  # 沿着第三个维度拼接

    phi_predict = np.reshape(model(input).detach().cpu().numpy(), [1, -1])
    phi_predict_fourier = np.reshape(
        np.fft.fft(phi_predict * np.sqrt(2 * np.pi) / M), [1, -1]
    )
    phi_predict_f = tn.resort_fourier(phi_predict_fourier, N)
    EDP = tn.EDP(M, N, para_f)
    u_predict = EDP.phi_to_pde(
        phi_predict_f,
        out_data,
    )
    # print(u_predict.shape)
    U_pred = tn.block(index, u_predict)

    u_true = EDP.phi_to_pde(
        phi_true_f,
        out_data,
    )
    U_true = tn.block(index, u_true)
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
    plt.colorbar()
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
    plt.colorbar()
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


if __name__ == "__main__":

    training_data_resolution = 128
    save_index = 0

    FNO_main(training_data_resolution, save_index)
