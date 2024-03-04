import os
from pathlib import Path

from config_parameters import *


# 处理我们想要训练数据集子集的情况
class RestrictedFilePathCreatorTorch:
    def __init__(self, ratio, all_file_paths, data_dir_path):
        self.ratio = ratio
        self.all_files_paths = all_file_paths
        self.data_dir_path = data_dir_path

        if ratio == 0.0:
            self.__create_file_names_from_existing_data()
        elif 0.0 < self.ratio < 1.0:
            self.__create_split_from_file_names()
        else:
            raise ValueError("Class should only be called when 0.0 <= ratio < 1.0")

    def __create_file_names_from_existing_data(self):
        # 将所有文件名扩展名调整为正确的扩展名
        all_file_names_with_correct_extension = self.__apply_correct_extension_to_set(
            self.all_files_paths
        )

        to_remove = []
        # 遍历所有带有正确扩展名的文件名
        for pos, line in enumerate(all_file_names_with_correct_extension):
            # 获取单个左图路径
            single_left_path = os.path.join(self.data_dir_path, line[0])

            if (
                single_left_path
                == "/home/rick/data/2011_09_28/2011_09_28_drive_0001_sync/image_02/data/0000000090.jpg"
            ):
                print("hey")
            # 如果左图路径不存在，则将其位置添加到待删除列表中
            if not os.path.exists(single_left_path):
                to_remove.append(pos)

        # 按照待删除位置列表逆序删除对应位置的文件名
        for pos in to_remove[::-1]:
            self.all_files_paths.pop(pos)

    # 根据给定的比例分割文件
    def __create_split_from_file_names(self):
        split_at = int(self.ratio * len(self.all_files_paths))
        self.all_files_paths = self.all_files_paths[:split_at]

    # 为文件名列表添加正确的文件扩展名
    def __apply_correct_extension_to_set(self, file_names):
        all_files = list(Path(self.data_dir_path).rglob("*.[pj][np][g]"))
        # 创建一个集合，用于存储所有文件的扩展名
        exts_set = {
            os.path.splitext(str(name))[1]
            for name in all_files
            # 排除特定文件夹中的文件
            if not any(
                substring in str(name)
                for substring in ["training", "testing", "cs_disparity"]
            )
        }
        # 如果包含.png，则为文件名列表中的每个文件名添加.png
        if "png" in exts_set:
            new_file_names = [
                [os.path.splitext(ele)[0] + ".png" for ele in line]
                for line in file_names
            ]
            return new_file_names
        return file_names


# 检查指定数据集中的所有图像是否存在于给定的数据目录中
def check_if_all_images_are_present(data_set, data_dir_path):
    set_names = ["train", "val", "test"]

    if data_set == "kitti":
        data_set_path = KITTI_PATH
    elif data_set == "eigen":
        data_set_path = EIGEN_PATH
    else:
        data_set_path = CITYSCAPES_PATH

    all_file_names_paths = []
    for set_name in set_names:
        full_path_to_set = data_set_path.format(set_name)
        with open(full_path_to_set, "r") as f:
            data_set_names = f.read()
            data_set_names = data_set_names.splitlines()
        for names in data_set_names:
            names = names.split(" ")
            all_file_names_paths.extend(names)

    total_length = len(all_file_names_paths)
    number_not_present = 0
    for file_path in all_file_names_paths:
        full_file_path = os.path.join(data_dir_path, file_path)
        if os.path.exists(full_file_path):
            number_not_present += 1

    print("-- Dir check for {}: --".format(data_set))
    print("{} out of {} images present".format(number_not_present, total_length))
    return


# 获取存在于指定数据目录中的图像路径
def get_present_images_from_list(lst_of_paths, data_dir):
    to_remove = []
    for pos, line in enumerate(lst_of_paths):
        line = line.split()
        single_left_path = os.path.join(data_dir, line[0])
        if not os.path.exists(single_left_path):
            to_remove.append(pos)

    for pos in to_remove[::-1]:
        lst_of_paths.pop(pos)
    return lst_of_paths
