import torch
import torch.optim as optim

from architectures import BaseArchitecture
from networks import (
    define_D,
    define_generator_loss,
    define_discriminator_loss,
    to_device,
)
from utils.image_pool import ImagePool


# class VanillaGanSingleArchitecture(BaseArchitecture):
class GanSingleArchitecture(BaseArchitecture):
    def __init__(self, args):
        super().__init__(args)

        if args.mode == "train":
            self.D = define_D(args)
            self.D = self.D.to(self.device)

            # # 创建一个用于存储虚假右侧图像的图像池，以降低模型过拟合的风险
            self.fake_right_pool = ImagePool(50)

            self.criterion = define_generator_loss(args)
            self.criterion = self.criterion.to(self.device)
            self.criterionGAN = define_discriminator_loss(args)
            self.criterionGAN = self.criterionGAN.to(self.device)

            self.optimizer_G = optim.Adam(self.G.parameters(), lr=args.learning_rate)
            self.optimizer_D = optim.SGD(self.D.parameters(), lr=args.learning_rate)

        # 根据模式加载正确的网络
        if args.mode == "train":
            self.model_names = ["G", "D"]
            self.optimizer_names = ["G", "D"]
        else:
            self.model_names = ["G"]

        self.loss_names = ["G", "G_MonoDepth", "G_GAN", "D"]
        self.losses = {}

        if self.args.resume:
            self.load_checkpoint()

        if "cuda" in self.device:
            torch.cuda.synchronize()

    def set_input(self, data):
        self.data = to_device(data, self.device)
        self.left = self.data["left_image"]
        self.right = self.data["right_image"]

    def forward(self):
        self.disps = self.G(self.left)

        # 生成器输出视差图像中提取的右视差通道
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in self.disps]
        self.disp_right_est = disp_right_est[0]

        # 使用生成器G生成虚假的右视图像
        # 使用左图像和右视差估计作为输入，以生成右图像
        self.fake_right = self.criterion.generate_image_right(
            self.left, self.disp_right_est
        )

    def backward_D(self):
        # 计算虚假数据的判别器损失
        fake_pool = self.fake_right_pool.query(self.fake_right)
        pred_fake = self.D(fake_pool.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # 计算真实数据的判别器损失
        pred_real = self.D(self.right)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # 计算生成器的对抗损失
        pred_fake = self.D(self.fake_right)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # 计算重建损失
        self.loss_G_MonoDepth = self.criterion(self.disps, [self.left, self.right])

        # 将对抗损失和视差损失加权求和作为总体损失
        self.loss_G = (
            self.loss_G_GAN * self.args.discriminator_w + self.loss_G_MonoDepth
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # 更新 D
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # 更新 G
        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_learning_rate(self, epoch, learning_rate):
        if self.args.adjust_lr:
            # 在 30 个 epoch 后，每过 10 个 epoch 学习率减半
            if 30 <= epoch < 40:
                lr = learning_rate / 2
            # 在 40 个 epoch 后学习率再减半
            elif epoch >= 40:
                lr = learning_rate / 4
            else:
                lr = learning_rate
            for param_group in self.optimizer_G.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer_D.param_groups:
                param_group["lr"] = lr

    def get_untrained_loss(self):
        # 计算生成器的损失，包括单视深度估计损失和生成器对抗损失
        loss_G_MonoDepth = self.criterion(self.disps, [self.left, self.right])
        fake_G_right = self.D(self.fake_right)
        loss_G_GAN = self.criterionGAN(fake_G_right, True)
        loss_G = loss_G_GAN * self.args.discriminator_w + loss_G_MonoDepth

        # 计算鉴别器的损失，包括对真实右视图和假右视图的鉴别损失
        loss_D_fake = self.criterionGAN(self.D(self.fake_right), False)
        loss_D_real = self.criterionGAN(self.D(self.right), True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return {
            "G": loss_G.item(),
            "G_MonoDepth": loss_G_MonoDepth.item(),
            "G_GAN": loss_G_GAN.item(),
            "D": loss_D.item(),
        }

    @property
    def architecture(self):
        return "Single GAN Architecture"
