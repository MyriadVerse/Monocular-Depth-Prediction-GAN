import os
import torch
import pickle
from networks import define_G, to_device


class BaseArchitecture:

    def __init__(self, args):
        self.args = args
        self._name = args.model_name
        self.resume = args.resume
        self.device = args.device

        self.output_directory = args.output_dir
        self.model_dir = args.model_dir

        self.model_store_path = os.path.join(args.model_dir, args.model_name)
        if not os.path.exists(self.model_store_path) and args.mode == "train":
            os.mkdir(self.model_store_path)
            # 同时将所有参数保存到一个文本文件中
            self.write_args_string_to_file()
        if not os.path.exists(self.model_store_path) and args.mode == "test":
            raise FileNotFoundError(
                "Model does not exist. Please check if the model has yet been run."
            )

        self.model_names = []
        self.loss_names = []
        self.optimizer_names = []

        self.losses = {}
        self.start_epoch = 0

        self.G = define_G(args)
        self.G = self.G.to(self.device)
        if args.use_multiple_gpu:
            self.G = torch.nn.DataParallel(self.G)
        print(
            "G [{}] initiated with {} trainable parameters".format(
                args.generator_model, self.num_parameters
            )
        )

        self.loader = None

    def fit(self, data, train_test=False):
        data = to_device(data, self.args.device)
        if not train_test:
            # 获取左图像数据并压缩维度
            left = data["left_image"].squeeze()
            # 输入左图像数据并返回生成的右图像数据
            return self.G(left)
        # 只有在训练时进行测试运行时才会执行以下代码。
        flipped_left = data["flipped_left_image"]
        left = data["left_image"]
        # 将左图像和翻转后的左图像堆叠起来，组成一个测试图像对
        test_image_pair = torch.stack((left, flipped_left)).squeeze()
        return self.G(test_image_pair)

    def set_data_loader(self, loader):
        # 当使用WGAN架构变体时，需要设置loader
        self.loader = loader

    def run_epoch(self, current_epoch, n_img):
        i = 0
        for data in self.loader:
            if i % 1000 == 0:
                print("Now run_epoch has been executed " + str(i) + "times")
            self.set_input(data)
            self.optimize_parameters()
            # 收集运行损失
            self.add_running_loss_train(current_epoch)
            i += 1

        # 估算每张图片的损失
        self.make_running_loss(current_epoch, n_img)
        print("In run_epoch, running loss is been made")

    def set_input(self, data):
        pass

    def optimize_parameters(self):
        pass

    def update_learning_rate(self, epoch, learning_rate):
        pass

    def get_untrained_loss(self):
        pass

    # 设置 requires_grad=False 以避免计算
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                # 遍历网络的每个参数
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # 为指定训练周期设置新的损失项
    def set_new_loss_item(self, epoch, train=True):
        if epoch not in self.losses:
            self.losses[epoch] = {}
        self.losses[epoch]["train" if train else "val"] = {
            name: 0.0 for name in self.loss_names
        }

    # 将当前训练步骤的损失添加到运行损失记录中
    def add_running_loss_train(self, epoch):
        for loss_name in self.loss_names:
            self.losses[epoch]["train"][loss_name] += getattr(
                self, "loss_{}".format(loss_name)
            ).item()

    # 将当前验证步骤的损失添加到运行损失记录中
    def add_running_loss_val(self, epoch):
        losses = self.get_untrained_loss()
        for loss_name in self.loss_names:
            self.losses[epoch]["val"][loss_name] += losses[loss_name]

    # 将累计的损失值转换为每个样本的平均损失
    def make_running_loss(self, epoch, n_img, train=True):
        if train:
            for loss_name in self.loss_names:
                self.losses[epoch]["train"][loss_name] /= n_img / self.args.batch_size
        # 验证阶段 batch 尺寸为1
        else:
            for loss_name in self.loss_names:
                self.losses[epoch]["val"][loss_name] /= n_img

    # 保存模型参数
    def save_networks(self, name):
        for model_name in self.model_names:
            if isinstance(model_name, str):
                save_filename = "{}_{}.pth".format(name, model_name)
                save_path = os.path.join(self.model_store_path, save_filename)

                net = getattr(self, model_name)
                torch.save(net.state_dict(), save_path)
                # 移除先前的检查点文件
                for file_name in os.listdir(self.model_store_path):
                    path_to_possible_checkpoint = os.path.join(
                        self.model_store_path, file_name
                    )
                    if "checkpoint" in file_name and os.path.isfile(
                        path_to_possible_checkpoint
                    ):
                        os.unlink(path_to_possible_checkpoint)

    def load_networks(self, name):
        for model_name in self.model_names:
            if isinstance(model_name, str):
                load_filename = "{}_{}.pth".format(name, model_name)
                load_path = os.path.join(self.model_store_path, load_filename)
                net = getattr(self, model_name)
                # 如果当前模型是 DataParallel 类型，则获取其模块部分
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print("loading the model from {}".format(load_path))
                # 加载模型的状态字典
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

    def save_best_networks(self):
        path_to_best_checkpoint = os.path.join(
            self.model_store_path, "model_best.pth.tar"
        )
        best_checkpoint = torch.load(path_to_best_checkpoint)

        for model_name in self.model_names:
            if isinstance(model_name, str):
                net = getattr(self, model_name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                # 更新模型参数为最佳模型的参数
                net.load_state_dict(best_checkpoint["state_dicts"][model_name])

                best_model_save_path = os.path.join(
                    self.model_store_path, "best_{}.pth".format(model_name)
                )
                torch.save(net.state_dict(), best_model_save_path)

        # 删除先前保存的检查点文件
        self.remove_checkpoints()

    def save_checkpoint(self, epoch, is_best, best_val_loss):
        # Retrieve a state.
        state = {
            "epoch": epoch + 1,
            "architecture": self.args.architecture,
            "state_dicts": {  # 记录所有模型的状态字典
                name: getattr(self, name).state_dict() for name in self.model_names
            },
            "best_val_loss": best_val_loss,
            "optimizers": {
                name: getattr(self, "optimizer_{}".format(name)).state_dict()
                for name in self.optimizer_names
            },
            "path": self.model_store_path,
            "name": self.name,
            "losses": self.losses,
        }

        # 删除先前保存的检查点文件
        self.remove_checkpoints()
        # 保存新的检查点文件
        filename = "checkpoint_e{}.pth.tar".format(state["epoch"] - 1)
        if is_best:
            filename = "model_best.pth.tar"
        full_save_path = os.path.join(self.model_store_path, filename)
        torch.save(state, full_save_path)

    def load_checkpoint(self, load_optim=True):
        if os.path.isfile(self.args.resume):
            print("=> loading checkpoint '{}'".format(self.args.resume))
            checkpoint = torch.load(self.args.resume)

            self.start_epoch = checkpoint["epoch"]
            self.model_store_path = checkpoint["path"]
            self._name = checkpoint["name"]
            self.losses = checkpoint["losses"]

            for model_name in self.model_names:
                if isinstance(model_name, str):
                    print(
                        "=> Now loading model and optimizer for model {}".format(
                            model_name
                        )
                    )
                    net = getattr(self, model_name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    net.load_state_dict(checkpoint["state_dicts"][model_name])

            if load_optim:
                for optimizer_name in self.optimizer_names:
                    optimizer = getattr(self, "optimizer_{}".format(optimizer_name))
                    optimizer.load_state_dict(checkpoint["optimizers"][optimizer_name])

            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(self.args.resume))

    def remove_checkpoints(self):
        for file_name in os.listdir(self.model_store_path):
            path_to_possible_checkpoint = os.path.join(self.model_store_path, file_name)
            if "checkpoint" in file_name and os.path.isfile(
                path_to_possible_checkpoint
            ):
                os.unlink(path_to_possible_checkpoint)

    def save_losses(self):
        path = os.path.join(
            self.output_directory, "final_losses_{}.p".format(self._name)
        )
        with open(path, "wb") as loss_output_file:
            pickle.dump(self.losses, loss_output_file)

    def to_test(self):
        for model_name in self.model_names:
            if isinstance(model_name, str):
                net = getattr(self, model_name)
                net.eval()

    def to_train(self):
        for model_name in self.model_names:
            if isinstance(model_name, str):
                net = getattr(self, model_name)
                net.train()

    def write_args_string_to_file(self):
        args_string = "=== ARGUMENTS ===\n"
        for key, val in vars(self.args).items():
            args_string += "{0: <20}: {1}\n".format(key, val)
        args_string += "================="
        with open(os.path.join(self.model_store_path, "params"), "w") as f:
            f.write(args_string)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.G.parameters() if p.requires_grad)

    @property
    def architecture(self):
        return "BaseArchitecture"

    @property
    def name(self):
        return self._name
