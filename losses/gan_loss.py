import torch
import torch.nn as nn


# 定义了使用 LSGAN 或常规 GAN 的 GAN 损失函数。
# 当使用 LSGAN 时，它基本上与 MSELoss 相同，
# 但它抽象出了创建与输入具有相同大小的目标标签张量的需要。
# 如果target_real_label != 1.0，而target_fake_label = 0.0
# 则应用单侧标签平滑化。
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        # 将真实标签和虚假标签注册为缓冲张量
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        # 根据是否使用LSGAN选择损失函数类型
        if use_lsgan:
            # 如果使用LSGAN，则损失函数为均方误差（MSELoss）
            self.loss = nn.MSELoss()
            print("Using LSGAN loss for the discriminator")
        else:
            # 否则，使用二元交叉熵损失（BCELoss）
            self.loss = nn.BCELoss()
            print("Using Vanilla BCE loss for the discriminator")

    # 根据是否为真实标签获取相应的目标张量
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    # 计算损失值
    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # 使用损失函数计算输入和目标之间的损失值
        return self.loss(input, target_tensor)
