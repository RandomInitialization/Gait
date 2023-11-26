import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h, w)


class block1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(block1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1)
        self.batch_normal = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normal(x)
        return self.relu(x)


class block2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(block2, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(5, 5), padding=2)
        self.batch_normal = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normal(x)
        return self.relu(x)


class block3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(block3, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, stride=(2, 2))
        self.batch_normal = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normal(x)
        return self.relu(x)


class block4(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(block4, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1, stride=(1, 1))
        self.batch_normal = nn.BatchNorm2d(out_channel)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_normal(x)
        x = self.relu(x)
        return self.pool(x)


class PCMAM(nn.Module):
    def __init__(self, in_channel):
        super(PCMAM, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        n, c, h, w = x.shape
        x1 = x.reshape(n, -1, c)  # h*w x c
        x2 = x.reshape(n, c, -1)  # c x h*w
        x11 = torch.matmul(x1, x2)  # h*w x h*w
        x12 = torch.matmul(x2, x11)  # c x h*w
        x13 = x12.reshape(n, c, h, w)

        x21 = torch.matmul(x2, x1)  # c x c
        x22 = torch.matmul(x1, x21)  # h*w x c
        x23 = x22.reshape(n, c, h, w)
        return x13 + x23


class SetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None
        _set_in_channels = 1
        _set_channels = [32, 64, 128]

        self.layer11 = block1(_set_in_channels, _set_channels[0])
        self.layer12 = block2(_set_in_channels, _set_channels[0])

        self.layer21 = block1(_set_channels[0], _set_channels[0])
        self.layer22 = block2(_set_channels[0], _set_channels[0])

        self.layer31 = block1(_set_channels[0], _set_channels[1])
        self.layer32 = block2(_set_channels[0], _set_channels[1])

        self.layer41 = block3(_set_channels[1], _set_channels[1])
        self.layer42 = block4(_set_channels[1], _set_channels[1])

        self.layer51 = block1(_set_channels[1], _set_channels[2])
        self.layer52 = block2(_set_channels[1], _set_channels[2])

        self.layer61 = block3(_set_channels[2], _set_channels[2])
        self.layer62 = block4(_set_channels[2], _set_channels[2])

        self.gl_layer31 = block1(_set_channels[0], _set_channels[1])
        self.gl_layer32 = block2(_set_channels[0], _set_channels[1])

        self.gl_layer41 = block3(_set_channels[1], _set_channels[1])
        self.gl_layer42 = block4(_set_channels[1], _set_channels[1])

        self.gl_layer51 = block1(_set_channels[1], _set_channels[2])
        self.gl_layer52 = block2(_set_channels[1], _set_channels[2])

        self.gl_layer61 = block3(_set_channels[2], _set_channels[2])
        self.gl_layer62 = block4(_set_channels[2], _set_channels[2])

        self.x_pcmam1 = PCMAM(_set_channels[0])
        self.x_pcmam2 = PCMAM(_set_channels[0])
        self.x_pcmam3 = PCMAM(_set_channels[1])
        self.x_pcmam4 = PCMAM(_set_channels[1])
        self.x_pcmam5 = PCMAM(_set_channels[2])
        self.x_pcmam6 = PCMAM(_set_channels[2])

        self.gl_pcmam3 = PCMAM(_set_channels[1])
        self.gl_pcmam4 = PCMAM(_set_channels[1])
        self.gl_pcmam5 = PCMAM(_set_channels[2])
        self.gl_pcmam6 = PCMAM(_set_channels[2])

        self.bin_num = [1, 2, 4, 8, 16]  # HPM的5个scale
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num) * 2, 128, hidden_dim)))])

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)  # 第二维度选最大
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
            ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

    def frame_median(self, x):
        if self.batch_frame is None:
            return torch.median(x, 1)  # 第二维度求平均
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
            ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list

    def forward(self, silho, batch_frame=None):
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum, :, :]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        n = silho.size(0)  # n=128
        x = silho.unsqueeze(2)  # 在第三维上增加一个维度也就是torch.Size([128, 30, 1, 64, 44])增加的应该是channel的维度
        del silho

        n, s, c, h, w = x.shape
        x = x.reshape(-1, c, h, w)
        x11 = self.layer11(x)
        x12 = self.layer12(x)
        x1 = x11 + x12
        x1 = self.x_pcmam1(x1)
        _, c, h, w = x1.shape
        x1 = x1.reshape(n, s, c, h, w)

        n, s, c, h, w = x1.shape
        x2 = x1.reshape(-1, c, h, w)
        x21 = self.layer21(x2)
        x22 = self.layer22(x2)
        x2 = x21 + x22
        x2 = self.x_pcmam2(x2)
        _, c, h, w = x2.shape
        x2 = x2.reshape(n, s, c, h, w)

        n, s, c, h, w = x2.shape
        x3 = x2.reshape(-1, c, h, w)  # _,32,64,64
        x31 = self.layer31(x3)
        x32 = self.layer32(x3)
        x3 = x31 + x32
        x3 = self.x_pcmam3(x3)
        _, c, h, w = x3.shape
        x3 = x3.reshape(n, s, c, h, w)  # _,64,64,64

        n, s, c, h, w = x3.shape
        x4 = x3.reshape(-1, c, h, w)
        x41 = self.layer41(x4)
        x42 = self.layer42(x4)
        x4 = x41 + x42
        x4 = self.x_pcmam4(x4)
        _, c, h, w = x4.shape
        x4 = x4.reshape(n, s, c, h, w)  # _,64,32,22

        n, s, c, h, w = x4.shape
        x5 = x4.reshape(-1, c, h, w)
        x51 = self.layer51(x5)
        x52 = self.layer52(x5)
        x5 = x51 + x52
        x5 = self.x_pcmam5(x5)
        _, c, h, w = x5.shape
        x5 = x5.reshape(n, s, c, h, w)

        n, s, c, h, w = x5.shape  # _,128,32,22
        x6 = x5.reshape(-1, c, h, w)
        x61 = self.layer61(x6)
        x62 = self.layer62(x6)
        x6 = x61 + x62
        x6 = self.x_pcmam6(x6)
        _, c, h, w = x6.shape
        x6 = x6.reshape(n, s, c, h, w)  # _,16,11
        x = self.frame_max(x6)[0]  # p*q,128,16,11

        gl = self.frame_max(x2)[0]
        gl31 = self.gl_layer31(gl)
        gl32 = self.gl_layer32(gl)
        gl3 = gl31 + gl32
        gl3 = self.gl_pcmam3(gl3)

        gl41 = self.gl_layer41(gl3)
        gl42 = self.gl_layer42(gl3)
        gl4 = gl41 + gl42
        gl4 = self.gl_pcmam4(gl4) + self.frame_max(x4)[0]

        gl51 = self.gl_layer51(gl4)
        gl52 = self.gl_layer52(gl4)
        gl5 = gl51 + gl52
        gl5 = self.gl_pcmam5(gl5)

        gl61 = self.gl_layer61(gl5)
        gl62 = self.gl_layer62(gl5)
        gl6 = gl61 + gl62
        gl6 = self.gl_pcmam6(gl6)

        gl = gl6 + x

        feature = list()
        n, c, h, w = gl.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
            z = gl.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc_bin[0])
        feature = feature.permute(1, 0, 2).contiguous()

        return feature, None
