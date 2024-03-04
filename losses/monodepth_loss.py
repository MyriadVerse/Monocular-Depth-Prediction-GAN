import torch
from .base_generator_loss import BaseGeneratorLoss


class MonodepthLoss(BaseGeneratorLoss):
    def __init__(self, args):
        super().__init__(args)
        # SSIM loss权重
        self.SSIM_w = args.alpha_image_loss_w

        self.img_loss_w = args.img_loss_w  # 图像损失权重
        self.disp_gradient_w = args.disp_grad_loss_w  # 视差梯度损失权重
        self.lr_w = args.lr_loss_w  # 左右一致性损失权重
        self.occl_w = args.occl_loss_w  # 遮挡损失权重

        self.n = args.num_disps  # 视差图金字塔尺度数量

    def forward(self, input, target):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        left, right = target
        # 将金字塔尺度调整为与视差图尺度相同
        left_pyramid = self.scale_pyramid(left, self.n)
        right_pyramid = self.scale_pyramid(right, self.n)

        # 准备视差图
        disp_left_est = [d[:, 0, :, :].unsqueeze(1) for d in input]
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in input]

        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est

        # 生成图像
        left_est = [
            self.generate_image_left(right_pyramid[i], disp_left_est[i])
            for i in range(self.n)
        ]
        right_est = [
            self.generate_image_right(left_pyramid[i], disp_right_est[i])
            for i in range(self.n)
        ]
        self.left_est = left_est
        self.right_est = right_est

        # L-R 一致性
        right_left_disp = [
            self.generate_image_left(disp_right_est[i], disp_left_est[i])
            for i in range(self.n)
        ]
        left_right_disp = [
            self.generate_image_right(disp_left_est[i], disp_right_est[i])
            for i in range(self.n)
        ]

        # 视差图平滑度
        disp_left_smoothness = self.disp_smoothness(disp_left_est, left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est, right_pyramid)

        # L1
        l1_left = [
            torch.mean(torch.abs(left_est[i] - left_pyramid[i])) for i in range(self.n)
        ]
        l1_right = [
            torch.mean(torch.abs(right_est[i] - right_pyramid[i]))
            for i in range(self.n)
        ]

        # SSIM
        ssim_left = [
            torch.mean(self.SSIM(left_est[i], left_pyramid[i])) for i in range(self.n)
        ]
        ssim_right = [
            torch.mean(self.SSIM(right_est[i], right_pyramid[i])) for i in range(self.n)
        ]

        image_loss_left = [
            self.SSIM_w * ssim_left[i] + (1 - self.SSIM_w) * l1_left[i]
            for i in range(self.n)
        ]
        image_loss_right = [
            self.SSIM_w * ssim_right[i] + (1 - self.SSIM_w) * l1_right[i]
            for i in range(self.n)
        ]
        image_loss = sum(image_loss_left + image_loss_right)

        # L-R Consistency
        lr_left_loss = [
            torch.mean(torch.abs(right_left_disp[i] - disp_left_est[i]))
            for i in range(self.n)
        ]
        lr_right_loss = [
            torch.mean(torch.abs(left_right_disp[i] - disp_right_est[i]))
            for i in range(self.n)
        ]
        lr_loss = sum(lr_left_loss + lr_right_loss)

        # Disparities smoothness
        disp_left_loss = [
            torch.mean(torch.abs(disp_left_smoothness[i])) / 2**i for i in range(self.n)
        ]
        disp_right_loss = [
            torch.mean(torch.abs(disp_right_smoothness[i])) / 2**i
            for i in range(self.n)
        ]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)

        # Occlusion
        occl_left_loss = [
            torch.mean(torch.abs(self.disp_left_est[i])) for i in range(self.n)
        ]
        occl_right_loss = [
            torch.mean(torch.abs(self.disp_right_est[i])) for i in range(self.n)
        ]
        occl_loss = sum(occl_left_loss + occl_right_loss)

        loss = (
            image_loss * self.img_loss_w
            + self.disp_gradient_w * disp_gradient_loss
            + self.lr_w * lr_loss
            + self.occl_w * occl_loss
        )

        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss
        self.occl_loss = occl_loss

        return loss
