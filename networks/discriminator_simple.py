import torch.nn as nn


class SimpleDiscriminator(nn.Module):
    def __init__(self, num_out=64):
        super(SimpleDiscriminator, self).__init__()
        self.num_out = num_out

        # 主体部分定义为一个包含多个卷积层的序列
        main = nn.Sequential(
            nn.Conv2d(3, self.num_out, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_out, 2 * self.num_out, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.num_out, 4 * self.num_out, 3, 2, padding=1),
            nn.LeakyReLU(),
        )
        self.main = main

        # 最后的线性全连接层，输入大小为4x4x4*num_out，输出大小为1
        self.linear = nn.Linear(4 * 4 * 4 * self.num_out, 1)

    def forward(self, input):
        output = self.main(input)
        # 将输出展平为一维向量
        output = output.view(-1, 4 * 4 * 4 * self.num_out)
        # 线性全连接层的前向传播
        output = self.linear(output)
        return output
