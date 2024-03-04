import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.bilinear_sampler import apply_disparity
from .ssim import ssim_gauss, ssim_godard


class BaseGeneratorLoss(nn.modules.Module):
    def __init__(self, args):
        super(BaseGeneratorLoss, self).__init__()
        self.which_ssim = args.which_ssim
        self.ssim_window_size = args.ssim_window_size

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = img.size()
        h = s[2]
        w = s[3]

        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(
                nn.functional.interpolate(
                    img, [nh, nw], mode="bilinear", align_corners=False
                )
            )
        return scaled_imgs

    def gradient_x(self, img):
        # 计算图像在 x 方向上的梯度
        # 使用镜像填充保持输出大小一致
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # 计算图像在 y 方向上的梯度
        # 使用镜像填充保持输出大小一致
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    # 使用视差信息生成左视图图像
    def generate_image_left(self, img, disp):
        return apply_disparity(img, -disp)

    # 使用视差信息生成右视图图像
    def generate_image_right(self, img, disp):
        return apply_disparity(img, disp)

    # 计算结构相似性指数（SSIM）
    def SSIM(self, x, y):
        if self.which_ssim == "godard":
            return ssim_godard(x, y)
        elif self.which_ssim == "gauss":
            return ssim_gauss(x, y, window_size=self.ssim_window_size)
        else:
            raise ValueError("{} version not implemented".format(self.which_ssim))

    # 计算视差图的平滑性
    def disp_smoothness(self, disp, pyramid):
        # 计算视差图在 x 和 y 方向上的梯度
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        # 计算图像金字塔中图像在 x 和 y 方向上的梯度
        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        # 计算 x 和 y 方向上的权重
        weights_x = [
            torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
            for g in image_gradients_x
        ]
        weights_y = [
            torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
            for g in image_gradients_y
        ]

        # 计算 x 和 y 方向上的平滑性
        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]

        return smoothness_x + smoothness_y

    def forward(self, input, target):
        pass
