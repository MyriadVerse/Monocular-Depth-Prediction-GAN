import os
from PIL import Image

from torch.utils.data import Dataset
from utils.reduce_image_set import RestrictedFilePathCreatorTorch


class StreetMonoLoader(Dataset):
    def __init__(self, data_dir, path_to_file_paths, mode, train_ratio, transform=None):
        with open(path_to_file_paths, "r") as f:
            all_paths = f.read().splitlines()
            all_split_paths = [pair.split(" ") for pair in all_paths]

        # 如果 train_ratio 不等于 1.0，则选择数据子集
        if train_ratio != 1.0:
            restricted_creator = RestrictedFilePathCreatorTorch(
                train_ratio, all_split_paths, data_dir
            )
            all_split_paths = restricted_creator.all_files_paths

        # 分离左右文件路径
        left_right_path_lists = [lst for lst in zip(*all_split_paths)]

        # 获取排序后的左图像路径
        left_fnames = list(left_right_path_lists[0])
        self.left_paths = sorted(
            [os.path.join(data_dir, fname) for fname in left_fnames]
        )

        # 如果是训练或验证模式，则获取排序后的右图像路径
        if mode == "train" or mode == "val":
            right_fnames = list(left_right_path_lists[1])
            self.right_paths = sorted(
                [os.path.join(data_dir, fname) for fname in right_fnames]
            )

            # 检查左右路径的数量是否相等
            assert len(self.right_paths) == len(
                self.left_paths
            ), "Paths file might be corrupted."

        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        # 尝试打开左图像
        try:
            left_image = Image.open(self.left_paths[idx])
            # 如果是训练模式，尝试打开右图像
            if self.mode == "train":
                right_image = Image.open(self.right_paths[idx])
                # 创建样本字典，包含左右图像
                sample = {"left_image": left_image, "right_image": right_image}
                # 如果有 transform，则应用到样本上
                if self.transform:
                    sample = self.transform(sample)
            # 如果是验证模式，同样需要左右图像以及测试样式的左图像翻转
            elif self.mode == "val":
                right_image = Image.open(self.right_paths[idx])
                sample = {"left_image": left_image, "right_image": right_image}
                sample = self.transform(sample)
            # 在测试时，只使用左图像和其翻转后的图像
            elif self.mode == "test":
                if self.transform:
                    left_image = self.transform(left_image)
                sample = {"left_image": left_image}
            else:
                raise ValueError("Mode {} not found in DataLoader".format(self.mode))
            return sample
        except FileNotFoundError:
            print(f"File at index {idx} does not exist. Skipping...")
            return None
