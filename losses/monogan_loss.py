import torch
from .base_generator_loss import BaseGeneratorLoss


class MonoganLoss(BaseGeneratorLoss):
    def __init__(self, args):
        super().__init__(args)
        # SSIM loss权重
        self.lamda = args.alpha_image_loss_w

        self.loss_ap_w = args.img_loss_w  # 图像损失权重
        self.loss_ds_w = args.disp_grad_loss_w  # 视差梯度损失权重
        self.loss_lr_w = args.lr_loss_w  # 左右一致性损失权重

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
            torch.mean(self.SSIM(left_est[i], left_pyramid[i])) / 2
            for i in range(self.n)
        ]
        ssim_right = [
            torch.mean(self.SSIM(right_est[i], right_pyramid[i])) / 2
            for i in range(self.n)
        ]

        # loss_ap
        image_loss_left = [
            (self.lamda * ssim_left[i] + (1 - self.lamda) * l1_left[i])
            for i in range(self.n)
        ]
        image_loss_right = [
            (self.lamda * ssim_right[i] + (1 - self.lamda) * l1_right[i])
            for i in range(self.n)
        ]
        loss_ap = sum(image_loss_left + image_loss_right)

        # 视差图平滑度
        disp_left_smoothness = self.disp_smoothness(disp_left_est, left_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_right_est, right_pyramid)

        # loss_ds
        disp_left_loss = [
            torch.mean(torch.abs(disp_left_smoothness[i])) / 2**i for i in range(self.n)
        ]
        disp_right_loss = [
            torch.mean(torch.abs(disp_right_smoothness[i])) / 2**i
            for i in range(self.n)
        ]
        loss_ds = sum(disp_left_loss + disp_right_loss)

        # L-R 一致性
        right_left_disp = [
            self.generate_image_left(disp_right_est[i], disp_left_est[i])
            for i in range(self.n)
        ]
        left_right_disp = [
            self.generate_image_right(disp_left_est[i], disp_right_est[i])
            for i in range(self.n)
        ]

        # loss_lr
        lr_left_loss = [
            torch.mean(torch.abs(right_left_disp[i] - disp_left_est[i]))
            for i in range(self.n)
        ]
        lr_right_loss = [
            torch.mean(torch.abs(left_right_disp[i] - disp_right_est[i]))
            for i in range(self.n)
        ]
        loss_lr = sum(lr_left_loss + lr_right_loss)

        loss = (
            self.loss_ap_w * loss_ap
            + self.loss_ds_w * loss_ds
            + self.loss_lr_w * loss_lr
        )

        self.loss_ap = loss_ap
        self.loss_ds = loss_ds
        self.loss_lr = loss_lr

        return loss
