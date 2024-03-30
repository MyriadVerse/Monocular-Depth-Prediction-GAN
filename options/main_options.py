from .base_options import BaseOptions, boolstr


class MainOptions(BaseOptions):

    def __init__(self):
        super(BaseOptions).__init__()

    def get_arguments(self, parser):
        parser = BaseOptions.get_arguments(self, parser)

        parser.add_argument(
            "--epochs", type=int, help="number of total epochs to run", default=50
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            help="initial learning rate (default: 1e-4)",
            default=1e-4,
        )
        parser.add_argument(
            "--adjust_lr",
            type=boolstr,
            help="apply learning rate decay or not",
            default=True,
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            help="Optimizer to use [adam | rmsprop]",
            default="adam",
        )
        parser.add_argument(
            "--batch_size", type=int, help="mini-batch size (default: 2)", default=2
        )

        parser.add_argument(
            "--generator_loss",
            type=str,
            help="generator loss: monodepth defaulted",
            default="monodepth",
        )
        parser.add_argument(
            "--discriminator_loss",
            type=str,
            help="generator loss [ vanilla | ls ]",
            default="ls",
        )
        parser.add_argument(
            "--img_loss_w", type=float, help="image reconstruction weight", default=1.0
        )
        parser.add_argument(
            "--lr_loss_w", type=float, help="left-right consistency weight", default=1.0
        )
        parser.add_argument(
            "--alpha_image_loss_w",
            type=float,
            help="weight between SSIM and L1 in the image loss",
            default=0.85,
        )
        # 用于计算结构相似性指数（SSIM）
        parser.add_argument(
            "--which_ssim",
            type=str,
            help="use either Godard SSIM or Gaussian [ godard | gauss ]",
            default="godard",
        )
        parser.add_argument(
            "--ssim_window_size",
            type=int,
            help="when using Gaussian SSIM, size of window",
            default=11,
        )
        parser.add_argument(
            "--disp_grad_loss_w",
            type=float,
            help="disparity smoothness weight",
            default=0.1,
        )
        parser.add_argument(
            "--occl_loss_w", type=float, help="Occlusion loss weight", default=0.0
        )
        parser.add_argument(
            "--discriminator_w",
            type=float,
            help="discriminator loss weight",
            default=0.0001,
        )
        parser.add_argument(
            "--do_augmentation",
            type=boolstr,
            help="do augmentation of images or not",
            default=True,
        )
        parser.add_argument(
            "--augment_parameters",
            type=str,
            help="lowest and highest values for gamma, brightness and color respectively",
            default=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
        )
        parser.add_argument(
            "--norm_layer",
            type=str,
            help="defines if a normalization layer is used",
            default="",
        )
        parser.add_argument(
            "--num_disps",
            type=int,
            help="Number of predicted disparity maps",
            default=4,
        )
        parser.add_argument(
            "--wgan_critics_num",
            type=int,
            help="number of critics in the WGAN architecture",
            default=5,
        )

        parser.add_argument(
            "--train_ratio",
            type=float,
            help="How much of the training data to use",
            default=1.0,
        )
        parser.add_argument(
            "--resume",
            type=str,
            help="path to latest checkpoint (default: none)",
            default="",
        )
        # 模型恢复训练时，G 和 D 的反向传播阶段次数
        parser.add_argument(
            "--resume_regime",
            type=str,
            help="back-passes of the G, D resp. default: 0,0",
            default=[0, 0],
        )
        # 每训练生成器一定次数后再更新判别器
        parser.add_argument(
            "--k",
            type=int,
            help=" the ratio between the number of training iterations performed on G and those on D",
            default=5,
        )

        parser.add_argument(
            "--postprocessing",
            type=boolstr,
            help="Do post-processing on depth maps",
            default=True,
        )
        parser.add_argument(
            "--load_final",
            type=boolstr,
            help="Load final or best trained model",
            default=True,
        )
        parser.add_argument(
            "--test_time",
            type=boolstr,
            help="Print the time of inference",
            default=True,
        )

        return parser
