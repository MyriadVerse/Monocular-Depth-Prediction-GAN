import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class Conv(nn.Module):
    def __init__(
        self, num_in_layers, num_out_layers, kernel_size, stride, normalize=None
    ):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        # 创建卷积层
        self.conv_base = nn.Conv2d(
            num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride
        )
        # 如果存在归一化层，则初始化归一化层
        if normalize:
            self.normalize = normalize(num_out_layers)
        else:
            self.normalize = normalize

    def forward(self, x):
        # 计算 padding 大小
        p = int(np.floor((self.kernel_size - 1) / 2))
        # 表示四个方向上的填充数量
        p2d = (p, p, p, p)
        # 使用pad函数对输入进行padding
        x = self.conv_base(F.pad(x, p2d))
        # 如果存在归一化层，则对输出进行归一化
        if self.normalize:
            x = self.normalize(x)
        # 使用ELU激活函数处理输出
        return F.elu(x, inplace=True)


# 定义一个卷积块类
class ConvBlock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, normalize=None):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv(
            num_in_layers, num_out_layers, kernel_size, 1, normalize=normalize
        )
        self.conv2 = Conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


# 最大池化层
class MaxPool(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size

    # 前向传播函数，将输入张量进行最大池化操作
    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


# 基于残差连接的卷积模块
class ResConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride, normalize=None):
        super(ResConv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride

        self.conv1 = Conv(num_in_layers, num_out_layers, 1, 1, normalize=normalize)
        self.conv2 = Conv(
            num_out_layers, num_out_layers, 3, stride, normalize=normalize
        )

        total_num_out_layers = 4 * num_out_layers
        self.conv3 = nn.Conv2d(
            num_out_layers, total_num_out_layers, kernel_size=1, stride=1
        )
        # 投影
        self.conv4 = nn.Conv2d(
            num_in_layers, total_num_out_layers, kernel_size=1, stride=stride
        )

        # 归一化层
        if normalize:
            self.normalize = normalize(total_num_out_layers)
        else:
            self.normalize = normalize

    def forward(self, x):
        # 判断是否需要投影层
        do_proj = x.size()[1] != self.num_out_layers or self.stride == 2

        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)

        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x

        # 归一化和残差连接
        if self.normalize:
            shortcut = self.normalize(x_out + shortcut)
        else:
            shortcut = x_out + shortcut

        # 使用ELU激活函数
        return F.elu(shortcut, inplace=True)


# ResNet18架构中使用的残差卷积块
class ResConvBasic(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride, normalize=None):
        super(ResConvBasic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride

        self.conv1 = Conv(num_in_layers, num_out_layers, 3, stride, normalize=normalize)
        self.conv2 = Conv(num_out_layers, num_out_layers, 3, 1, normalize=normalize)
        # 投影层
        self.conv3 = nn.Conv2d(
            num_in_layers, num_out_layers, kernel_size=1, stride=stride
        )

        if normalize:
            self.normalize = normalize(num_out_layers)
        else:
            self.normalize = normalize

    def forward(self, x):
        do_proj = x.size()[1] != self.num_out_layers or self.stride == 2

        x_out = self.conv1(x)
        x_out = self.conv2(x_out)

        # 投影
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x

        # 归一化和残差
        if self.normalize:
            shortcut = self.normalize(x_out + shortcut)
        else:
            shortcut = x_out + shortcut

        return F.elu(shortcut, inplace=True)


# 构建一个 ResNet 块
def ResBlock(num_in_layers, num_out_layers, num_blocks, stride, normalize=None):
    layers = []
    # 添加第一个 ResConv 层，步长为 stride
    layers.append(ResConv(num_in_layers, num_out_layers, stride, normalize=normalize))
    # 添加 num_blocks - 1 个 ResConv 层，步长为 1
    for i in range(1, num_blocks - 1):
        layers.append(
            ResConv(4 * num_out_layers, num_out_layers, 1, normalize=normalize)
        )
    # 添加最后一个 ResConv 层，步长为 1
    layers.append(ResConv(4 * num_out_layers, num_out_layers, 1, normalize=normalize))
    return nn.Sequential(*layers)


# 构建一个用于 ResNet18的 ResNet 块，
def ResBlockBasic(num_in_layers, num_out_layers, num_blocks, stride, normalize=None):
    layers = []
    # 添加第一个 ResConvBasic 层，步长为 stride
    layers.append(
        ResConvBasic(num_in_layers, num_out_layers, stride, normalize=normalize)
    )
    # 添加 num_blocks - 1 个 ResConvBasic 层，步长为 1
    for i in range(1, num_blocks):
        layers.append(
            ResConvBasic(num_out_layers, num_out_layers, 1, normalize=normalize)
        )
    return nn.Sequential(*layers)


# 上采样特征图所使用的上卷积层
# 对输入的特征图进行双线性插值上采样
# 然后通过卷积操作进一步处理
class Upconv(nn.Module):
    def __init__(
        self,
        num_in_layers,
        num_out_layers,
        kernel_size,
        scale,  # scale (int): 上采样比例
        normalize=None,
    ):
        super(Upconv, self).__init__()
        self.scale = scale
        self.conv1 = Conv(
            num_in_layers, num_out_layers, kernel_size, 1, normalize=normalize
        )

    def forward(self, x):
        # 对输入张量进行上采样
        x = nn.functional.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=True
        )
        return self.conv1(x)


# 定义一个获取视差的模块
class GetDisp(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=2, normalize=None):
        super(GetDisp, self).__init__()
        # 创建一个卷积层用于提取特征
        self.conv1 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=3, stride=1)

        if normalize:
            self.normalize = normalize(num_out_layers)
        else:
            self.normalize = normalize

        # 创建一个Sigmoid激活函数
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        if self.normalize:
            x = self.normalize(x)
        # 返回视差值（乘以0.3并通过Sigmoid激活函数）
        return 0.3 * self.sigmoid(x)


# 定义 UNet 中的跳跃连接模块
# 将底层特征与上采样路径中的特征进行连接
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc,  # 外部特征图通道数
        inner_nc,  # 内部特征图通道数
        input_nc=None,  # 输入特征图通道数，默认为外部特征图通道数
        submodule=None,  # 子模块（内部的 UnetSkipConnectionBlock）
        outermost=False,  # 是否为最外层
        innermost=False,  # 是否为最内层
        norm_layer=nn.BatchNorm2d,  # 标准化层，默认为批标准化
        # 是否使用 dropout, 随机地将一部分神经元的输出置为0
        use_dropout=False,
    ):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        # 如果归一化层是一个偏函数
        # 检查是否是用于创建 InstanceNorm2d 的归一化层
        # 将 use_bias 设置为 True 或 False
        # 以决定是否在卷积层中使用偏置项
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # 如果输入通道数未指定，则默认为外部通道数
        if input_nc is None:
            input_nc = outer_nc

        # 定义下采样卷积层，用于将输入特征图尺寸减半
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )

        # LeakyReLU激活函数，用于下采样后的特征图
        downrelu = nn.LeakyReLU(0.2, True)
        # 下采样后的特征图归一化层
        downnorm = norm_layer(inner_nc)
        # ReLU激活函数，用于上采样后的特征图
        uprelu = nn.ReLU(True)
        # 上采样后的特征图归一化层
        upnorm = norm_layer(outer_nc)

        # 如果是最外层块，使用反卷积将特征图放大
        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            # 输出前使用Tanh激活函数
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        # 如果是最内层块，只进行特征图放大
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            # 输出前使用ReLU激活函数
            up = [uprelu, upconv, upnorm]
            model = down + up
        # 中间层的块，进行特征图的放大和连接
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            # 使用dropout正则化
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # 如果是最外层模块，则直接将输入数据传递给模型进行处理
        if self.outermost:
            return self.model(x)
        # 如果不是，则对输入数据和模型处理后的结果进行拼接，并返回
        else:
            return torch.cat([x, self.model(x)], 1)


class FuseBlock(nn.Module):
    def __init__(self):
        super(FuseBlock, self).__init__()
        # 卷积层，用于融合两个输入通道为一个输出通道
        self.fuse_conv = Conv(2, 1, 1, 1)

    def forward(self, x):
        return self.fuse_conv(x)


class Self_Attn(nn.Module):
    """Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        # 定义查询、键、值的卷积层
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  # 注意力权重的 softmax

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        # 通过查询、键、值卷积层获取查询、键、值
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C x (*W*H)
        # 计算能量分数
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        # 计算注意力权重
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C X N

        # 计算自注意力值
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        # 加权融合输入特征和自注意力值，并应用缩放参数 gamma
        out = self.gamma * out + x
        return out, attention
