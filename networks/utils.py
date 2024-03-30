import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


from .generator_vggnet import VggNetMD, VggNetSuper

from .discriminator_nlayer import NLayerDiscriminator
from .discriminator_simple import SimpleDiscriminator

from losses import *

from config_parameters import D_CONV_FILTERS


# 根据给定的 norm_type 参数返回归一化层
def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=True
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


# 根据给定的优化器和参数配置返回学习率调度器
def get_scheduler(optimizer, opt):
    if opt.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(
                opt.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opt.lr_policy
        )
    return scheduler


# 初始化神经网络权重
def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            # 初始化卷积层和全连接层的权重
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            # 初始化偏置项
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # 初始化批归一化层的权重和偏置项
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


# 初始化神经网络
def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_D(args, n_layers=5, init_type="normal", init_gain=0.02, gpu_ids=[]):
    model = args.discriminator_model
    use_sigmoid = False if args.discriminator_loss == "ls" else True

    input_channels = args.disc_input_channels
    ndf = D_CONV_FILTERS

    # 根据是否有指定归一化层，确定使用的归一化层类型
    norm = "none" if not args.norm_layer else args.norm_layer
    norm_layer = get_norm_layer(norm_type=norm)

    if model == "simple":
        netD = SimpleDiscriminator()
    elif model == "n_layers":
        netD = NLayerDiscriminator(
            input_channels,
            ndf,
            n_layers,
            norm_layer=norm_layer,
            use_sigmoid=use_sigmoid,
        )
    else:
        raise NotImplementedError(
            "Discriminator model name {} is not recognized".format(model)
        )

    print("Now initializing {} discriminator.".format(model))
    return init_net(netD, init_type, init_gain, gpu_ids)


def define_G(args):
    model = args.generator_model
    input_channels = args.input_channels
    num_output_disparities = args.num_disp_channels

    # 根据指定的归一化层类型，确定使用的归一化类
    if args.norm_layer == "batch":
        normalize = nn.BatchNorm2d
    elif args.norm_layer == "instance":
        normalize = nn.InstanceNorm2d
    else:
        normalize = None

    pretrained = args.pretrained

    if model == "vgg":
        out_model = VggNetMD(
            input_channels, num_out_layers=num_output_disparities, normalize=normalize
        )
    elif model == "vgg_super":
        out_model = VggNetSuper(
            input_channels, num_out_layers=num_output_disparities, normalize=normalize
        )
    else:
        raise NotImplementedError(
            "Generator model {} is not implemented".format(args.generator_model)
        )
    print(
        "Now initializing {} generator using {} normalization".format(
            model, "no" if not normalize else args.norm_layer
        )
    )
    return out_model


def define_generator_loss(args):
    if args.generator_loss == "monodepth":
        loss = MonoganLoss(args)
    else:
        raise NotImplementedError(
            "Generator loss {} is not implemented".format(args.generator_loss)
        )
    return loss


def define_discriminator_loss(args):
    if args.discriminator_loss == "ls":
        loss = GANLoss()
    elif args.discriminator_loss == "vanilla":
        loss = GANLoss(use_lsgan=False, target_real_label=0.9)
    else:
        raise NotImplementedError(
            "Discriminator loss {} is not implemented".format(args.discriminator_loss)
        )
    return loss
