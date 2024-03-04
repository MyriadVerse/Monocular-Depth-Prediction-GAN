#
# Alwyn Mathew
# Monodepth in pytorch(https://github.com/alwynmathew/monodepth-pytorch)
# Bilinear sampler in pytorch(https://github.com/alwynmathew/bilinear-sampler-pytorch)
#

import torch
from torch.nn.functional import pad


# 将视差应用于输入图像以生成新图像
def apply_disparity(
    input_images,
    x_offset,
    wrap_mode="border",
    tensor_type=torch.cuda.FloatTensor,
    cuda_device="cuda:0",
):
    # 获取输入图像的维度信息
    num_batch, num_channels, height, width = input_images.size()
    device = input_images.device

    # 处理不同的边界模式
    edge_size = 0
    if wrap_mode == "border":
        edge_size = 1
        # 在最后两个维度两侧填充1个像素
        input_images = pad(input_images, (1, 1, 1, 1))
    elif wrap_mode == "edge":
        edge_size = 0
    else:
        return None

    # 将通道维度放到最慢的维度，将批次维度与其他维度展平
    input_images = input_images.permute(1, 0, 2, 3).contiguous()
    im_flat = input_images.reshape(num_channels, -1)

    # 创建像素索引的网格（PyTorch没有专门的meshgrid函数）
    x = (
        torch.linspace(0, width - 1, width)
        .repeat(height, 1)
        .type(tensor_type)
        .to(device=device)
    )
    y = (
        torch.linspace(0, height - 1, height)
        .repeat(width, 1)
        .transpose(0, 1)
        .type(tensor_type)
        .to(device=device)
    )
    # 考虑填充的影响
    x = x + edge_size
    y = y + edge_size
    # 展平并重复每个图像的像素索引
    x = x.reshape(-1).repeat(1, num_batch)
    y = y.reshape(-1).repeat(1, num_batch)

    # 现在我们要在 X 方向上采样偏移的像素索引
    # 为此，我们将视差从百分比转换为像素，并添加到 X 索引中
    x = x + x_offset.contiguous().reshape(-1) * width
    # 确保不超出图像范围
    x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
    # 将视差四舍五入到整数像素网格以进行采样
    y0 = torch.floor(y)
    # 在X方向上分别四舍五入到下和上，以便稍后进行线性插值
    x0 = torch.floor(x)
    x1 = x0 + 1
    # 向上取整后，可能再次超出图像边界
    x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

    # 计算从图像批次的展平版本中绘制的索引
    dim2 = width + 2 * edge_size
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    # 设置每个图像在批次中的偏移量
    base = dim1 * torch.arange(num_batch).type(tensor_type).to(device=device)
    base = base.reshape(-1, 1).repeat(1, height * width).view(-1)
    # Y 方向上的一个像素移动相当于在展平数组中移动 dim2
    base_y0 = base + y0 * dim2
    # 单独添加两个X方向上的偏移量
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    # 从图像中采样像素
    pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
    pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

    # 应用线性插值以考虑分数偏移量
    weight_l = x1 - x
    weight_r = x - x0
    output = weight_l * pix_l + weight_r * pix_r

    # 将输出重塑回图像批次并重新排列为（N,C,H,W）形状
    output = output.reshape(num_channels, num_batch, height, width).permute(1, 0, 2, 3)

    return output
