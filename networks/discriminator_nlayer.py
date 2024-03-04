import torch.nn as nn
import functools


# 定义具有指定参数的 PatchGAN 鉴别器
# 针对图像的局部判别器架构
# 将输入图像映射到一个标量值，表示图像的真假
class NLayerDiscriminator(nn.Module):
    def __init__(
        self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d, use_sigmoid=False
    ):
        super(NLayerDiscriminator, self).__init__()

        # 判断是否为偏置归一化层
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 3
        stride = 2
        pad_width = 1

        # 创建卷积层序列
        sequence = [
            nn.Conv2d(
                input_nc, ndf, kernel_size=kernel_size, stride=stride, padding=pad_width
            ),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        # 循环创建多个卷积层
        # 每层的输出通道数是上一层的 nf_mult 的两倍，但不超过 8 倍
        for n in range(1, n_layers - 2):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            # 如果存在归一化层
            if norm_layer:
                sequence += [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=pad_width,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                sequence += [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=pad_width,
                        bias=use_bias,
                    ),
                    nn.LeakyReLU(0.2, True),
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)

        if norm_layer:
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=pad_width,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        else:
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=pad_width,
                    bias=use_bias,
                ),
                nn.LeakyReLU(0.2, True),
            ]

        # 添加最后一个卷积层，输出通道数为 1，不包含归一化和激活函数
        sequence += [
            nn.Conv2d(
                ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=pad_width
            )
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
