import torch
from torch import nn
from torch.nn.functional import interpolate


class GloRe(nn.Module):
    def __init__(self, in_channels):
        super(GloRe, self).__init__()
        self.N = in_channels // 4
        self.S = in_channels // 2

        self.theta = nn.Conv2d(in_channels, self.N, 1, 1, 0, bias=False)
        self.phi = nn.Conv2d(in_channels, self.S, 1, 1, 0, bias=False)

        self.relu = nn.ReLU()

        self.node_conv = nn.Conv1d(self.N, self.N, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.S, self.S, 1, 1, 0, bias=False)

        # このunitに入力された時のチャンネル数と合わせるためのconv layer
        self.conv_2 = nn.Conv2d(self.S, in_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W

        B = self.theta(x).view(-1, self.N, L)

        phi = self.phi(x).view(-1, self.S, L)
        phi = torch.transpose(phi, 1, 2)

        V = torch.bmm(B, phi) / L  # 著者コード中にある謎割り算
        V = self.relu(self.node_conv(V))
        V = self.relu(self.channel_conv(torch.transpose(V, 1, 2)))

        y = torch.bmm(torch.transpose(B, 1, 2), torch.transpose(V, 1, 2))
        y = y.view(-1, self.S, H, W)
        y = self.conv_2(y)

        return x + y

class FCNHead(nn.Module):
    def __init__(self, in_channels, image_size, num_class, use_glore=True):
        super(FCNHead, self).__init__()
        self.image_size = image_size

        inter_channels = in_channels // 4
        self.conv51 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.use_glore = use_glore
        if self.use_glore:
            self.gcn = GloRe(inter_channels)

        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv53 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.Dropout2d(0.2),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, num_class, 3, padding=1, bias=False))

    def forward(self, x, image_size):
        x = self.conv51(x)
        if self.use_glore:
            x = self.gcn(x)
        x = self.conv52(x)
        x = interpolate(x, image_size)
        x = self.conv53(x)
        # x = x[:, :, 1:-1, 1:-1] # conv53のpaddingで拡大してしまった分を除去

        output = self.conv6(x)

        return output
