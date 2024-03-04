import torch
import torch.optim as optim
import math

from architectures import BaseArchitecture
from networks import define_D, define_generator_loss, to_device


LAMBDA = 10  # 梯度惩罚的超参数 lambda


# 实际上是一个 WGAN-GP 架构
class WGanArchitecture(BaseArchitecture):

    def __init__(self, args):
        super().__init__(args)

        if args.mode == "train":
            self.D = define_D(args)
            self.D = self.D.to(self.device)

            self.criterionMonoDepth = define_generator_loss(args)
            self.criterionMonoDepth = self.criterionMonoDepth.to(self.device)

            self.optimizer_G = optim.Adam(self.G.parameters(), lr=args.learning_rate)
            self.optimizer_D = optim.SGD(self.D.parameters(), lr=args.learning_rate)

            # 定义常数项
            self.one = torch.tensor(1.0).to(self.device)
            self.mone = (self.one * -1).to(self.device)

            self.loader_iterator = None
            self.current_epoch = 0
            # 定义每个 D 更新周期的迭代次数
            self.critic_iters = args.wgan_critics_num

        # 根据模式加载正确的网络
        if args.mode == "train":
            self.model_names = ["G", "D"]
            self.optimizer_names = ["G", "D"]
        else:
            self.model_names = ["G"]

        self.loss_names = ["G", "G_MonoDepth", "G_GAN", "D", "D_Wasserstein"]
        self.losses = {}

        if self.args.resume:
            self.load_checkpoint()

        if "cuda" in self.device:
            torch.cuda.synchronize()

    def run_epoch(self, current_epoch, n_img):
        self.loader_iterator = iter(self.loader)
        self.current_epoch = current_epoch

        # 计算需要运行的迭代次数（passes）
        passes = int(math.floor(n_img / self.args.batch_size / (self.critic_iters + 1)))

        for i in range(passes):
            self.optimize_parameters()
        # 估计每张图像的损失
        loss_divider = int(math.floor(n_img / (self.critic_iters + 1)))
        self.make_running_loss(current_epoch, loss_divider)

    def set_input(self, data):
        self.data = to_device(data, self.device)
        self.left = self.data["left_image"]
        self.right = self.data["right_image"]

    def forward(self):
        self.disps = self.G(self.left)

        # 准备视差图
        disp_right_est = [d[:, 1, :, :].unsqueeze(1) for d in self.disps]
        self.disp_right_est = disp_right_est[0]

        # 生成右视图
        self.fake_right = self.criterionMonoDepth.generate_image_right(
            self.left, self.disp_right_est
        )

    def backward_D(self):
        # 计算真实图像的判别器输出及损失
        right_var = torch.autograd.Variable(self.right)
        D_real = self.D(right_var)
        D_real = D_real.mean()
        D_real.backward(self.mone)

        # 计算生成的假图像的判别器输出及损失
        fake_right_var = torch.autograd.Variable(self.fake_right)
        D_fake = self.D(fake_right_var.detach())
        D_fake = D_fake.mean()
        D_fake.backward(self.one)

        # 计算梯度惩罚
        gradient_penalty = self.calculate_gradient_penalty(
            right_var.data, fake_right_var.data
        )
        gradient_penalty.backward()

        # 设置判别器损失和Wasserstein损失
        self.loss_D = D_fake - D_real + gradient_penalty
        self.loss_D_Wasserstein = D_real - D_fake

        # 在主循环之外设置损失，因为判别器的训练次数比生成器多
        self.losses[self.current_epoch]["train"]["D"] += (
            self.loss_D.item() / self.critic_iters
        )
        self.losses[self.current_epoch]["train"]["D_Wasserstein"] += (
            self.loss_D_Wasserstein.item() / self.critic_iters
        )

    def backward_G(self):
        # G 应该欺骗 D
        fake_right_var = torch.autograd.Variable(self.fake_right, requires_grad=True)
        self.loss_G_GAN = self.D(fake_right_var)
        self.loss_G_GAN = self.loss_G_GAN.mean() * self.args.discriminator_w
        self.loss_G_GAN.backward(self.mone)

        # 重构损失
        self.loss_G_MonoDepth = self.criterionMonoDepth(
            self.disps, [self.left, self.right]
        )
        self.loss_G_MonoDepth.backward()

        # 计算总体生成器损失
        self.loss_G = self.loss_G_GAN + self.loss_G_MonoDepth

        # 更新训练过程中的损失记录
        self.losses[self.current_epoch]["train"]["G"] += self.loss_G.item()
        self.losses[self.current_epoch]["train"]["G_GAN"] += self.loss_G_GAN.item()
        self.losses[self.current_epoch]["train"][
            "G_MonoDepth"
        ] += self.loss_G_MonoDepth.item()

    def optimize_parameters(self):
        # 更新 D
        self.set_requires_grad(self.D, True)
        for critic_iter in range(self.critic_iters):
            data = next(self.loader_iterator)
            self.set_input(data)
            self.forward()

            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # 更新 G
        data = next(self.loader_iterator)
        self.set_input(data)
        self.forward()

        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_learning_rate(self, epoch, learning_rate):
        if self.args.adjust_lr:
            # 每 10 个 epoch 减少一半，从第 30 个 epoch 开始
            if 30 <= epoch < 40:
                lr = learning_rate / 2
            elif epoch >= 40:
                lr = learning_rate / 4
            else:
                lr = learning_rate
            for param_group in self.optimizer_G.param_groups:
                param_group["lr"] = lr
            for param_group in self.optimizer_D.param_groups:
                param_group["lr"] = lr

    def get_untrained_loss(self):
        # 计算生成器的损失
        loss_G_MonoDepth = self.criterionMonoDepth(self.disps, [self.left, self.right])

        fake_right_var_G = torch.autograd.Variable(self.fake_right)
        fake_G = self.D(fake_right_var_G)
        loss_G_GAN = fake_G.mean() * self.args.discriminator_w

        loss_G = loss_G_GAN + loss_G_MonoDepth

        # 计算判别器的损失
        self.optimizer_D.zero_grad()

        fake_right_var_D = torch.autograd.Variable(self.fake_right)
        real_right_var_D = torch.autograd.Variable(self.right)

        fake_D = self.D(fake_right_var_D)
        loss_D_fake = fake_D.mean()
        real_D = self.D(real_right_var_D)
        loss_D_real = real_D.mean()
        # 计算梯度惩罚项
        gradient_penalty = self.calculate_gradient_penalty(
            real_right_var_D.data, fake_right_var_D.data, training=False
        )

        # 判别器总损失为假图片损失减去真图片损失
        loss_D = loss_D_fake - loss_D_real
        # 判别器 Wasserstein 损失
        loss_D_Wasserstein = loss_D_real - loss_D_fake + gradient_penalty

        return {
            "G": loss_G.item(),
            "G_MonoDepth": loss_G_MonoDepth.item(),
            "G_GAN": loss_G_GAN.item(),
            "D": loss_D.item(),
            "D_Wasserstein": loss_D_Wasserstein.item(),
        }

    def calculate_gradient_penalty(self, real, fake, training=True):
        # 生成随机数作为插值系数
        alpha = torch.rand(self.args.batch_size, 1)
        alpha = (
            alpha.expand(
                self.args.batch_size, int(real.nelement() / self.args.batch_size)
            )
            .contiguous()
            .view(real.size())
        )
        alpha = alpha.to(self.device)

        # 生成插值
        interpolates = alpha * real + ((1 - alpha) * fake)
        interpolates = interpolates.to(self.device)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        # 计算插值处的判别结果
        disc_interpolates = self.D(interpolates)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
            create_graph=training,
            retain_graph=training,
            only_inputs=True,
        )
        gradients = gradients[0]
        gradients = gradients.view(gradients.size(0), -1)

        # 计算梯度惩罚项
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        # 下面是一种实现方法，用于解决计算梯度惩罚时的数值不稳定性
        # https://github.com/pytorch/pytorch/issues/2534
        # 对我而言，设置 epsilon 会稍微有影响，但仍然可能导致梯度爆炸。
        # epsilon = 1e-5
        # gradient_penalty = torch.mean((1. - torch.sqrt(epsilon + torch.sum(gradients.view(gradients.size(0), -1) ** 2, dim=1))) ** 2)

        return gradient_penalty

    @property
    def architecture(self):
        return "WGAN-GP Architecture"
