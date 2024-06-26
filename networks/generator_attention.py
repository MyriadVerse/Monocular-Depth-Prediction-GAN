from networks.blocks import *


class VggNetMD(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=2, normalize=None):
        super(VggNetMD, self).__init__()
        # encoder
        self.conv1 = ConvBlock(num_in_layers, 32, 7, normalize=normalize)  # H/2
        self.conv2 = ConvBlock(32, 64, 5, normalize=normalize)  # H/4
        self.conv3 = ConvBlock(64, 128, 3, normalize=normalize)  # H/8
        self.conv4 = ConvBlock(128, 256, 3, normalize=normalize)  # H/16
        self.conv5 = ConvBlock(256, 512, 3, normalize=normalize)  # H/32
        self.conv6 = ConvBlock(512, 512, 3, normalize=normalize)  # H/64
        self.conv7 = ConvBlock(512, 512, 3, normalize=normalize)  # H/128

        # 添加注意力机制
        self.attn1 = Self_Attn(32, "relu")
        self.attn2 = Self_Attn(64, "relu")
        self.attn3 = Self_Attn(128, "relu")
        self.attn4 = Self_Attn(256, "relu")
        self.attn5 = Self_Attn(512, "relu")
        self.attn6 = Self_Attn(512, "relu")
        self.attn7 = Self_Attn(512, "relu")

        # decoder
        self.upconv7 = Upconv(512, 512, 3, 2, normalize=normalize)
        self.iconv7 = Conv(512 + 512, 512, 3, 1, normalize=normalize)

        self.upconv6 = Upconv(512, 512, 3, 2, normalize=normalize)
        self.iconv6 = Conv(512 + 512, 512, 3, 1, normalize=normalize)

        self.upconv5 = Upconv(512, 256, 3, 2, normalize=normalize)
        self.iconv5 = Conv(256 + 256, 256, 3, 1, normalize=normalize)

        self.upconv4 = Upconv(256, 128, 3, 2, normalize=normalize)
        self.iconv4 = Conv(128 + 128, 128, 3, 1, normalize=normalize)
        self.disp4_layer = GetDisp(128, num_out_layers=num_out_layers)

        self.upconv3 = Upconv(128, 64, 3, 2, normalize=normalize)
        self.iconv3 = Conv(64 + 64 + num_out_layers, 64, 3, 1, normalize=normalize)
        self.disp3_layer = GetDisp(64, num_out_layers=num_out_layers)

        self.upconv2 = Upconv(64, 32, 3, 2, normalize=normalize)
        self.iconv2 = Conv(32 + 32 + num_out_layers, 32, 3, 1, normalize=normalize)
        self.disp2_layer = GetDisp(32, num_out_layers=num_out_layers)

        self.upconv1 = Upconv(32, 16, 3, 2, normalize=normalize)
        self.iconv1 = Conv(16 + num_out_layers, 16, 3, 1, normalize=normalize)
        self.disp1_layer = GetDisp(16, num_out_layers=num_out_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x1, p1 = self.attn1(x1)

        x2 = self.conv2(x1)
        x2, p2 = self.attn2(x2)

        x3 = self.conv3(x2)
        x3, p3 = self.attn3(x3)

        x4 = self.conv4(x3)
        x4, p4 = self.attn4(x4)

        x5 = self.conv5(x4)
        x5, p5 = self.attn5(x5)

        x6 = self.conv6(x5)
        x6, p6 = self.attn6(x6)

        x7 = self.conv7(x6)
        x7, p7 = self.attn7(x7)

        # skips
        skip1 = x1
        skip2 = x2
        skip3 = x3
        skip4 = x4
        skip5 = x5
        skip6 = x6

        # decoder
        upconv7 = self.upconv7(x7)
        concat7 = torch.cat((upconv7, skip6), 1)
        iconv7 = self.iconv7(concat7)

        upconv6 = self.upconv6(iconv7)
        concat6 = torch.cat((upconv6, skip5), 1)
        iconv6 = self.iconv6(concat6)

        upconv5 = self.upconv5(iconv6)
        concat5 = torch.cat((upconv5, skip4), 1)
        iconv5 = self.iconv5(concat5)

        upconv4 = self.upconv4(iconv5)
        concat4 = torch.cat((upconv4, skip3), 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        self.udisp4 = nn.functional.interpolate(
            self.disp4, scale_factor=2, mode="bilinear", align_corners=True
        )

        upconv3 = self.upconv3(iconv4)
        concat3 = torch.cat((upconv3, skip2, self.udisp4), 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        self.udisp3 = nn.functional.interpolate(
            self.disp3, scale_factor=2, mode="bilinear", align_corners=True
        )

        upconv2 = self.upconv2(iconv3)
        concat2 = torch.cat((upconv2, skip1, self.udisp3), 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        self.udisp2 = nn.functional.interpolate(
            self.disp2, scale_factor=2, mode="bilinear", align_corners=True
        )

        upconv1 = self.upconv1(iconv2)
        concat1 = torch.cat((upconv1, self.udisp2), 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)

        return (
            self.disp1,
            self.disp2,
            self.disp3,
            self.disp4,
            p1,
            p2,
            p3,
            p4,
            p5,
            p6,
            p7,
        )
