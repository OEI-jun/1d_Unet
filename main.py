import torch
import torchvision.transforms.functional
from torch import nn
import time
import numpy as np
import matplotlib as plt
import torch.utils.data as data
from scipy.io import loadmat
from scipy.io import savemat

batch_size = 32
num_epochs = 100
shuffle = False
lr = 1e-3
weight_decay = 0


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.first = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same')
        self.BN1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU()

        self.second = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding='same')
        self.BN2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        x = self.BN1(x)
        x = self.act1(x)
        x = self.second(x)
        x = self.act2(x)
        x = self.BN2(x)
        return x


class DownSample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):

    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        # contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2]])
        x = torch.cat([x, contracting_x], dim=1)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 64), (64, 128), (128, 256), (256, 512)]])

        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        self.middle_cov = DoubleConvolution(512, 1024)

        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64)]])

        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])

        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        pass_through = []

        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)

        x = self.middle_cov(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)

        x = self.final_conv(x)
        return x


# 网络实例化
net = UNet(1, 1)
# 损失函数
criterion = nn.MSELoss()
# 优化函数
# optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)


# 训练数据读取
def data_load(file, data_name):
    data = loadmat(file, mat_dtype=True)
    train_data = data[data_name]
    train_data = train_data.astype(np.float32)
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
    train_data = torch.from_numpy(train_data)
    return train_data


x_dat_train = data_load('E:\\科研资料\\一维弹簧模型\\光声激发\\train_data_X_Unet.mat', 'train_data_X_Unet')
y_dat_train = data_load('E:\\科研资料\\一维弹簧模型\\光声激发\\train_data_y_Unet.mat', 'train_data_y_Unet')
train_data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_dat_train, y_dat_train),
                                         batch_size=batch_size, shuffle=shuffle)

# 是否使用GPU加速
use_cuda = torch.cuda.is_available()
if use_cuda:
    net = net.cuda()


# 训练网络
def train(net, train_iter, loss, num_epochs, optimizer, use_cuda):
    loss_train_list = []
    time_start = time.time()

    for epoch in range(num_epochs):
        for X, y in train_iter:
            if use_cuda:
                X = X.cuda()
                y = y.cuda()
            y_hat = net(X)
            total_loss = loss(y_hat, y).mean()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        time_end = time.time()
        loss_train_list.append(total_loss.item())
        print('epoch %d, train_loss %.10f, time %d s' % (epoch + 1, total_loss.item(), time_end - time_start))

    return loss_train_list


loss_list = train(net, train_data, criterion, num_epochs, optimizer, use_cuda)
plt.figure()
plt.plot(loss_list)
plt.show()