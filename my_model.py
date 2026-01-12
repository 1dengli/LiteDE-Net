from torchvision import models as resnet_model
from modules2 import dffm
from modules2 import gfmm
from modules2 import maem
from modules2 import resblock
from torch import nn
import torch
from thop import profile



class m88(nn.Module):
    def __init__(self, channel):
        super(m88, self).__init__()
        self.con1 = nn.Conv2d(channel, 3, 1)


        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.as1 = maem.maem(3, 64)
        self.as2 = maem.maem(64, 128)
        self.as3 = maem.maem(128, 256)
        self.as4 = maem.maem(256, 512)

        self.cf1 = dffm.dffm(64)
        self.cf2 = dffm.dffm(128)
        self.cf3 = dffm.dffm(256)
        self.cf4 = dffm.dffm(512)


        self.max_p = nn.MaxPool2d(2, stride=2)

        self.tran = gfmm.gfmm(512, 8, 8)


        self.up6 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv6 = resblock.DoubleConv(512, 256)
        self.up7 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv7 = resblock.DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv8 = resblock.DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv10 = nn.Conv2d(32, 1, 1)
        self.dropout = nn.Dropout2d(p=0.5)

        self.out_up = nn.ConvTranspose2d(64, 1, 2, 2)



    def forward(self, x):

        x = self.con1(x)
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)  # 64 64 64


        e1 = self.encoder1(e0)    # 64 64 642
        e2 = self.encoder2(e1)    # 128 32 32
        e3 = self.encoder3(e2)     # 256 16 16
        e4 = self.encoder4(e3)  # 512 8 8

        x_a = self.max_p(x)
        as1 = self.as1(x_a)
        cf1 = self.cf1([as1, e1])
        max1 = self.max_p(cf1)

        as2 = self.as2(max1)
        cf2 = self.cf2([as2, e2])
        max2 = self.max_p(cf2)

        as3 = self.as3(max2)
        cf3 = self.cf3([as3, e3])
        max3 = self.max_p(cf3)

        as4 = self.as4(max3)
        cf4 = self.cf4([as4, e4])
        mid1 = self.dropout(cf4)

        tran = self.tran(mid1)

        mid2 = self.dropout(tran)

        up_6 = self.up6(mid2)
        merge6 = torch.cat([up_6, cf3], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, cf2], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, cf1], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        c10 = self.conv10(up_9)

        e_out = self.out_up(e0)

        return c10, e_out




if __name__ == '__main__':
    unet = m88(1)
    unet.eval()  # Forward shape

    rgb = torch.randn([1, 1, 128, 128])

    out1, out2= unet(rgb)

    # FLOPs & Params
    flops, params = profile(unet, inputs=(rgb,))

    flop_g = flops / 1e9  # 转为 GFLOPs
    param_num = params  # 参数数量
    param_size_mb = params * 4 / (1024 * 1024)  # 4 bytes per float32

    print(f"模型的 FLOPs：{flop_g:.3f} GFLOPs")
    print(f"参数数量：{param_num}")
    print(f"模型大小：{param_size_mb:.3f} MB")
