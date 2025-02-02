"""
Modified based on  Zongyi Li's code

"""

import random
import sys
from matplotlib.ticker import MaxNLocator
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
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

# import test_Numerical as tn
import os
import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer

current_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_path, "../../src/")

if src_path in sys.path:
    sys.path.remove(src_path)

sys.path.insert(0, src_path)

from utilities3 import *
import test_Numerical as tn

# import scipy


print("\n=============================")
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(): " + str(torch.cuda.get_device_name()))
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
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)

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
                [256, 560, 560, 560, 560],
                [128, 560, 560, 560, 560],
                [1, 560, 560, 560, 560],
            ]
        )

    def forward(self, x):

        xd = self.don(x)
        x = torch.cat([xd.reshape(xd.shape[0], xd.shape[1], 1), x[:, :, 3:]], dim=2)
        # x = xd.reshape(xd.shape[0], xd.shape[1], 1)
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
        for i in range(len(self.Layers_branch2) - 1):
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

    batch_size = 2**13
    learning_rate = 0.001

    epochs = 5000  # default 500
    step_size = 500  # default 100

    gamma = 0.5

    modes = 10
    width = 128
    print(
        f"Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}, step_size:{step_size}, lr_decay: {gamma}"
    )
    ################################################################
    # read training data
    ################################################################
    torch.cuda.set_device(1)
    # Data is of the shape (number of samples, grid size)
    path = "/home/ext8/mengbin/BI-TDONet_update/LBVP/data_exterior_Neumann_problem"
    words = os.path.basename(path).split("_")[-3:]
    problem = "".join(word[0] for word in words).upper()
    print(problem)

    start_time = time.time()

    data_path = path + "/data.mat"
    print("Data location:", data_path)
    dataloader = MatReader(data_path)
    para = dataloader.read_field("para")
    phi = dataloader.read_field("phi")
    f = dataloader.read_field("f")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data reading time: {elapsed_time:.4f} seconds")
    m = f.shape[0]
    random.seed(1117)
    idx = np.array(range(m))
    random.shuffle(idx)
    para = para[idx, :]
    phi = phi[idx, :]
    f = f[idx, :]

    M = 128
    N = (f.shape[1] - 1) // 2
    name = f"LBVP_BI-DeepFNO_{problem}_0111"
    print(name)
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

    print("Shape of Training Dataset :", x_train.shape, y_train.shape)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=y_test.shape[0] // 100,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
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

    traced_model = torch.jit.trace(
        model.eval(), torch.tensor(x_test[0:2, :], dtype=torch.float32).cuda()
    )
    jit_model = torch.jit.freeze(traced_model)
    print("JIT Model loaded!")
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
            batch_size = x.shape[0]
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
                # out = model(x)
                out = jit_model(x)
                batch_size = x.shape[0]
                # out = y_normalizer.decode(out.view(batch_size, -1))
                test_l2 += myloss(out, y).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        train_losses.append((train_mse, train_l2))
        test_losses.append(test_l2)

        t2 = default_timer()
        # scheduler.step(test_l2)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "Epoch: %d, time: %.3f, Train Loss: %.4e,  Test l2: %.4e, lr: %.4e,"
            % (ep, t2 - t1, train_l2, test_l2, current_lr)
        )
        # print(ep, t2-t1, train_mse, train_l2, test_l2)

    elapsed = time.time() - start_time

    # Save the model state dictionary
    os.makedirs(checkpoint_save_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        "model/%s/%s" % (name, name),
    )

    print("\n=============================")
    print("Training done...")
    print("Total number of parameters: ", total_params)
    print("Training time: %.3f" % (elapsed))
    print("=============================\n")

    # ====================================
    # saving settings
    # ====================================

    current_directory = os.getcwd()
    resolution = "TrainRes_" + str(train_data_res)
    folder_index = str(save_index)

    results_dir = "/results/" + resolution + "/" + folder_index + "/"
    save_results_to = current_directory + results_dir
    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)
    model_dir = "/model/" + resolution + "/" + folder_index + "/"
    save_models_to = current_directory + model_dir
    if not os.path.exists(save_models_to):
        os.makedirs(save_models_to)
    # Save loss data to a CSV file
    loss_data = pd.DataFrame(train_losses, columns=["Train MSE", "Train L2"])
    loss_data["Test L2"] = test_losses
    loss_data.to_csv(f"loss/{name}.csv", index=False)

    loss_data = pd.read_csv(f"loss/{name}.csv")
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
        np.reshape(np.log10(loss_data["Train L2"]).ravel(), [-1, 1]),
        linewidth=5.0,
        label="BI-DeepFNO \n Training loss",
    )
    plt.plot(
        np.reshape(np.log10(loss_data["Test L2"].ravel()), [-1, 1]),
        linewidth=5.0,
        label="BI-DeepFNO \n Testing loss",
    )
    plt.xlabel("Epoch", fontweight="bold")
    plt.ylabel("lg(MRE)", fontweight="bold")
    plt.legend()
    plt.title("BI-DeepFNO Loss", fontweight="bold")
    # Set axis properties
    ax = plt.gca()  # Get the current axis object
    ax.yaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # Ensure Y-axis has integer tick marks
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

    # ====================================
    # testing
    # ====================================

    x_test = x_test
    y_test = y_test

    batch = 100
    total_size = x_test.shape[0]
    batch_size = total_size // batch

    errors, errors1, tt = [], [], []

    # Ensure the last batch is handled when the dataset size is not divisible by batch_size
    for i in range(
        0, len(x_test), batch_size
    ):  # Loop through the dataset with step size of batch_size
        end_idx = min(i + batch_size, len(x_test))  # Prevent out-of-bounds indexing
        x_batch, y_batch = x_test[i:end_idx], y_test[i:end_idx]
        x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
        start_time = time.time()
        pred_batch = model(x_batch)
        tt.append(time.time() - start_time)

        pred_batch = torch.squeeze(pred_batch, axis=2)

        mae = torch.mean(torch.norm(pred_batch - y_batch, dim=1))
        errors1.append(mae.item())
        errors.append(torch.mean(mae / torch.norm(y_batch, dim=1)).item())

    # Output average inference time
    avg_inference_time = np.sum(tt) / m1 * 1000
    print(f"Average inference time = {avg_inference_time:.4e} milliseconds")

    # Calculate average error and variance
    average_error1 = np.mean(errors1)
    variance1 = np.var(errors1)
    print(
        f"Average l2 error and variance across all batches: {average_error1:.4e}, {variance1:.4e}"
    )

    average_error = np.mean(errors)
    variance = np.var(errors)
    print(
        f"Average l2  relative error and variance across all batches: {average_error:.4e}, {variance:.4e}"
    )

    # ====================================
    # Example
    # ====================================
    t = np.linspace(0, 2 * np.pi, M, endpoint=False)
    t1 = np.linspace(0, 2 * np.pi, M)
    r1 = random.randint(0, m1)
    r2 = random.randint(0, m1)

    para = np.reshape(para_test[r1].numpy(), [1, -1])
    phi_true_f = np.reshape(phi_test[r1].numpy(), [1, -1])
    f_f = np.reshape(f_test[r1].numpy(), [1, -1])
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
    phi_true_f = np.reshape(phi_true_f, [1, -1])
    start = time.time()
    input = x_test[r1 : r1 + 1].cuda()
    phi_predict = model(input)
    phi_predict = phi_predict.detach().cpu().numpy().reshape(1, -1)
    phi_predict_fourier = np.reshape(
        np.fft.fft(phi_predict * np.sqrt(2 * np.pi) / M), [1, -1]
    )
    phi_predict_f = tn.resort_fourier(phi_predict_fourier, N)
    end = time.time()

    # This makes the text bold)
    # plt.rcParams["font.family"] = "serif"

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
    plt.title("The output of BI-DeepFNO", font={"size": 25})
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整
    # plt.savefig(
    #     f"../Figures/{name}_phi_1.eps",
    #     format="eps",
    # )
    # plt.savefig(
    #     f"../Figures/{name}_phi_1.pdf",
    #     format="pdf",
    # )

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(f_f, t1), [-1, 1]),
        linewidth=5.0,
    )
    plt.title(r"$\widetilde{f}(t)$", font={"size": 25})
    plt.xlabel(r"$t$", font={"size": 25})
    plt.ylabel(r"$\widetilde{f}(t)$", font={"size": 25})
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))

    plt.plot(
        np.reshape(tn.to_point(px, t1), [-1, 1]),
        np.reshape(tn.to_point(py, t1), [-1, 1]),
        linewidth=5.0,
    )
    plt.xlabel(r"$\widetilde{{\gamma}}_1(t)$", font={"size": 25})
    plt.ylabel(r"$\widetilde{{\gamma}}_2(t)$", font={"size": 25})
    plt.title("boundary", font={"size": 25})
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(phi_predict, [-1, 1])
        - np.reshape(tn.to_point(phi_true_f, t), [-1, 1]),
        label="Error",
    )
    plt.title("Error of BI-DeepFNO")
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整

    print("Model Time is:", (end * 1000 - start * 1000))
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
    # u_predict =  L1.phi_to_pde(phi_predict, out_data)
    u_true = L.phi_to_pde(phi_true_f, out_data)
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
    plt.pcolormesh(X, Y, U_true, cmap="jet", shading="gouraud")  # 彩虹热力图
    # plt.contourf(X,Y,Z_true)
    plt.colorbar()
    plt.xlabel(r"$x$", font={"size": 25})
    plt.ylabel(r"$y$", font={"size": 25})
    plt.title("True", font={"size": 25})
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        U_pred,
        cmap="jet",
        shading="gouraud",
    )  # 彩虹热力图
    # plt.contourf(X,Y,Z_true)
    plt.xlabel(r"$x$", font={"size": 25})
    plt.ylabel(r"$y$", font={"size": 25})
    plt.title("Predict", font={"size": 25})
    plt.colorbar()
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整
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
    )  # 彩虹热力图
    # plt.contourf(X,Y,Z_true)
    plt.xlabel(r"$x$", font={"size": 25})
    plt.ylabel(r"$y$", font={"size": 25})
    plt.title("Error", font={"size": 25})
    plt.colorbar()
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整
    # plt.savefig(
    #     f"../Figures/{name}_error_1.pdf",
    #     format="pdf",
    # )

    para = np.reshape(para_test[r2].numpy(), [1, -1])
    phi_true_f = np.reshape(phi_test[r2].numpy(), [1, -1])
    f_f = np.reshape(f_test[r2].numpy(), [1, -1])
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

    start = time.time()
    input = x_test[r2 : r2 + 1].cuda()
    phi_predict = model(input)
    phi_predict = phi_predict.detach().cpu().numpy().reshape(1, -1)
    phi_predict_fourier = np.reshape(
        np.fft.fft(phi_predict * np.sqrt(2 * np.pi) / M), [1, -1]
    )
    phi_predict_f = tn.resort_fourier(phi_predict_fourier, N)
    end = time.time()
    print(phi_predict_f)
    print(phi_true_f)

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
    plt.title("The output of BI-DeepFNO", font={"size": 25})
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整
    # plt.savefig(
    #     f"../Figures/{name}_phi_2.eps",
    #     format="eps",
    # )
    # plt.savefig(
    #     f"../Figures/{name}_phi_2.pdf",
    #     format="pdf",
    # )
    plt.figure(figsize=(8, 6))

    plt.plot(
        np.reshape(t1, [-1, 1]),
        np.reshape(tn.to_point(f_f, t1), [-1, 1]),
        linewidth=5.0,
    )
    plt.title(r"$\widetilde{f}(t)$", font={"size": 25})
    plt.xlabel(r"$t$", font={"size": 25})
    plt.ylabel(r"$\widetilde{f}(t)$")
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(tn.to_point(px, t1), [-1, 1]),
        np.reshape(tn.to_point(py, t1), [-1, 1]),
        linewidth=5.0,
    )
    plt.xlabel(r"$\widetilde{{\gamma}}_1(t)$", font={"size": 25})
    plt.ylabel(r"$\widetilde{{\gamma}}_2(t)$", font={"size": 25})
    plt.title("boundary", font={"size": 25})
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.plot(
        np.reshape(phi_predict, [-1, 1])
        - np.reshape(tn.to_point(phi_true_f, t), [-1, 1]),
        label="Error",
    )
    plt.title("Error of BI-DeepFNO")
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整

    print("Model Time is:", (end * 1000 - start * 1000))
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

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, U_true, cmap="jet", shading="gouraud")  # 彩虹热力图
    # plt.contourf(X,Y,Z_true)
    plt.colorbar()
    plt.xlabel(r"$x$", font={"size": 25})
    plt.ylabel(r"$y$", font={"size": 25})
    plt.title("True", font={"size": 25})
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        U_pred,
        cmap="jet",
        shading="gouraud",
    )  # 彩虹热力图
    # plt.contourf(X,Y,Z_true)
    plt.xlabel(r"$x$", font={"size": 25})
    plt.ylabel(r"$y$", font={"size": 25})
    plt.title("Predict", font={"size": 25})
    plt.colorbar()
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整
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
    )  # 彩虹热力图
    # plt.contourf(X,Y,Z_true)
    plt.xlabel(r"$x$", font={"size": 25})
    plt.ylabel(r"$y$", font={"size": 25})
    plt.title("Error", font={"size": 25})
    plt.colorbar()
    plt.tight_layout(pad=0.2)  # 使用 tight_layout 自动调整
    # plt.savefig(
    #     f"../Figures/{name}_error_2.pdf",
    #     format="pdf",
    # )
    plt.show()


if __name__ == "__main__":

    training_data_resolution = 128
    save_index = 0

    FNO_main(training_data_resolution, save_index)
