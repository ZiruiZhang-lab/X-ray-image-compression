import torch.nn as nn
import torch

class Structprior2(nn.Module):
    def __init__(self, in_channels, out_channels, num_scales=2, groups=1):
        super().__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        # x
        self.dwconv1_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=(5,2), groups=groups, bias=False, dilation=(5,2))
        self.dwconv2_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=(2,5), groups=groups, bias=False, dilation=(2,5))
        self.dwconv3_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, groups=groups, bias=False, dilation=2)
        # x_flip
        self.dwconv1_x_flip = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=(5,2), groups=groups, bias=False, dilation=(5,2))
        self.dwconv2_x_flip = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=(2,5), groups=groups, bias=False, dilation=(2,5))
        self.dwconv3_x_flip = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, groups=groups, bias=False, dilation=2)
        # self.Avgpool = nn.AvgPool2d(kernel_size = 2)
        self.Avgpool1 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=2)
        self.Avgpool2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=2)

        self.Maxpool = nn.MaxPool2d(kernel_size = 2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.Conv_down_x = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=2, padding=1)
        self.Conv_down_x_flip = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=2, padding=1)
        self.Conv_total_down = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=(2,1), padding=0)

        # encoder
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ) for _ in range(num_scales)
            ])
        self.conv_up =  nn.Conv2d(in_channels=192, out_channels=320, kernel_size=3, padding=1, stride=2)
        self.fuse = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        # output: [1, 192, 128, 128] -> [1, 192, 64, 64] -> [1, 192, 32, 32]
        x_flip = torch.flip(x, dims=[-1])  # [1, 1, 1024, 1024]
        x_res = self.Conv_down_x(x)
        x_flip_res =self.Conv_down_x_flip(x_flip)
        feats = []
        x1 = self.relu(self.dwconv1_x(x))
        x2 = self.relu(self.dwconv2_x(x))
        x3 = self.relu(self.dwconv3_x(x))

        x_cat = torch.cat([x1,x2,x3],dim=1)

        x_cat_avg = self.Avgpool1(x_cat)
        x_cat_max = self.Maxpool(x_cat)



        x_cat_max_sig = self.sigmoid(x_cat_max)

        x_cat_max_sig_res = x_cat_max_sig * x_res       # res[1,1,1024,1024]    sig[1,96,512,512]
        x_sum = x_cat_max_sig_res + x_cat_avg

        # flip
        x_flip1 = self.relu(self.dwconv1_x(x_flip))
        x_flip2 = self.relu(self.dwconv2_x(x_flip))
        x_flip3 = self.relu(self.dwconv3_x(x_flip))

        x_flip_cat = torch.cat([x_flip1,x_flip2,x_flip3],dim=1)

        x_flip_cat_avg = self.Avgpool2(x_flip_cat)
        x_flip_cat_max = self.Maxpool(x_flip_cat)
        x_flip_cat_sig = self.sigmoid(x_flip_cat_max)

        x_flip_cat_sig_res = x_flip_cat_sig * x_flip_res
        x_flip_sum = x_flip_cat_sig_res + x_flip_cat_avg

        feat = torch.cat([x_sum, x_flip_sum], dim=1)     # [1, 192, 512, 512]
        feat = self.fuse(feat)
        feats.append((feat))
        # feat = self.Conv_total_down(x_total)

        # fus           x_total[1,192,256,256]
        for i in range(self.num_scales):
            feat = self.encoders[i](feat)
            feats.append(feat)

        feat = self.relu(self.conv_up(feat))
        feats.append(feat)

        return feats  # list of [B, out_channels, H', W']



if __name__ == "__main__":

    x  = torch.randn([1, 1, 256, 256])
    struct = Structprior2(1 , 32)
    x_out = struct(x)
    for i in range(4):
        print(x_out[i].shape)



