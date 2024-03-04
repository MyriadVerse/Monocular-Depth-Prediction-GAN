import argparse
import ast
from pathlib import Path
import os


# 将字符串转换为布尔值
def boolstr(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


class BaseOptions:

    def __init__(self):
        pass

    def get_arguments(self, parser):
        parser.add_argument(
            "--architecture",
            type=str,
            help="The architecture to run.",
            default="gan_single",
        )
        parser.add_argument(
            "--mode", type=str, help="mode: train or test", default="train"
        )
        parser.add_argument(
            "--split", type=str, help="data split, kitti or eigen", default="eigen"
        )
        parser.add_argument(
            "--dataset",
            type=str,
            help="dataset to train on kitti or cityscapes or both",
            default="kitti",
        )

        parser.add_argument(
            "--data_dir", type=str, help="path to the data directory", required=True
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            help="where to save the disparities",
            default="output/",
        )
        parser.add_argument(
            "--model_dir",
            type=str,
            help="path to the trained models",
            default="saved_models/",
        )
        parser.add_argument(
            "--model_name", type=str, help="model name", default="model_default_name"
        )

        parser.add_argument(
            "--input_height", type=int, help="input height", default=256
        )
        parser.add_argument("--input_width", type=int, help="input width", default=512)
        parser.add_argument(
            "--generator_model",
            type=str,
            help="encoder architecture: vgg defaulted",
            default="vgg",
        )
        parser.add_argument(
            "--discriminator_model",
            type=str,
            help="discrininator architecture: [simple | n_layers]",
            default="n_layers",
        )

        parser.add_argument(
            "--pretrained",
            type=boolstr,
            help="use weights of pretrained model",
            default=False,
        )
        #  输入张量中的通道数
        parser.add_argument(
            "--input_channels",
            type=int,
            help="Number of channels in input tensor",
            default=3,
        )
        # 判别器输入张量中的通道数
        parser.add_argument(
            "--disc_input_channels",
            type=int,
            help="Number of channels in input tensor for discriminator",
            default=3,
        )
        # 视差通道数
        parser.add_argument(
            "--num_disp_channels",
            type=int,
            help="Number of disparity channels",
            default=2,
        )

        parser.add_argument(
            "--device", type=str, help='choose cpu or cuda:0 device"', default="cuda:0"
        )
        parser.add_argument(
            "--use_multiple_gpu",
            type=boolstr,
            help="whether to use multiple GPUs",
            default=False,
        )
        parser.add_argument(
            "--num_threads",
            type=int,
            help="number of threads to use for data loading",
            default=8,
        )

        return parser

    @staticmethod
    def print_options(args):
        print("=== ARGUMENTS ===")
        for key, val in vars(args).items():
            print("{0: <20}: {1}".format(key, val))
        print("=================")

    def parse(self):
        parser = argparse.ArgumentParser(description="Monodepth PyTorch implementation")
        parser = self.get_arguments(parser)

        args = parser.parse_args()

        # e.g. --data_dir ~/data
        if args.data_dir[0] == "~":
            home_path_string = str(Path.home())
            args.data_dir = os.path.join(home_path_string, args.data_dir[2:])

        # Wasserstein GAN.
        if args.architecture == "wgan":
            args.discriminator_model = "simple"

        # 将某些项转换为列表形式
        if hasattr(args, "resume_regime") and not type(args.resume_regime) == list:
            args.resume_regime = ast.literal_eval(args.resume_regime)
        if (
            hasattr(args, "augment_parameters")
            and not type(args.augment_parameters) == list
        ):
            args.augment_parameters = ast.literal_eval(args.augment_parameters)

        self.print_options(args)

        return args
