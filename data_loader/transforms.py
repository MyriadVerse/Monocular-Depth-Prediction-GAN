import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as f
import numpy as np


def image_transforms_kitti(
    mode="train",
    augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
    do_augmentation=True,
    transformations=None,
    size=(256, 512),
):
    if mode == "train":
        data_transform = transforms.Compose(
            [
                ResizeImage(train=True, size=size),
                RandomFlip(do_augmentation),
                ToTensor(train=True),
                AugmentImagePair(augment_parameters, do_augmentation),
            ]
        )
        return data_transform
    elif mode == "val":
        data_transform = transforms.Compose(
            [ResizeImage(train=True, size=size), ToTensor(train=True), DoVal()]
        )
        return data_transform
    elif mode == "test":
        data_transform = transforms.Compose(
            [
                ResizeImage(train=False, size=size),
                ToTensor(train=False),
                DoTest(),
            ]
        )
        return data_transform
    # 如果模式是自定义模式
    elif mode == "custom":
        data_transform = transforms.Compose(transformations)
        return data_transform
    else:
        print("Wrong mode")


def image_transforms_cityscapes(
    mode="train",
    augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
    do_augmentation=True,
    transformations=None,
    size=(256, 512),
):
    if mode == "train":
        data_transform = transforms.Compose(
            [
                CutLowPartofImage(train=True),
                ResizeImage(train=True, size=size),
                RandomFlip(do_augmentation),
                ToTensor(train=True),
                AugmentImagePair(augment_parameters, do_augmentation),
            ]
        )
        return data_transform
    elif mode == "val":
        data_transform = transforms.Compose(
            [
                CutLowPartofImage(train=True),
                ResizeImage(train=True, size=size),
                ToTensor(train=True),
                DoVal(),
            ]
        )
        return data_transform
    elif mode == "test":
        data_transform = transforms.Compose(
            [
                CutLowPartofImage(train=False),
                ResizeImage(train=False, size=size),
                ToTensor(train=False),
                DoTest(),
            ]
        )
        return data_transform
    elif mode == "custom":
        data_transform = transforms.Compose(transformations)
        return data_transform
    else:
        print("Wrong mode")


class CutLowPartofImage(object):
    """
    In CityScapes crop the upper 50 pixels, because of weird artifacts.
                  crop the bottom ~250 pixels to deal with the Mercedes sign of the car.
    """

    def __init__(self, train):
        self.train = train

    def __call__(self, sample):
        if self.train:
            left_image = sample["left_image"]
            right_image = sample["right_image"]
            new_left_image = f.crop(left_image, 50, 274, 750, 1500)
            new_right_image = f.crop(right_image, 50, 274, 750, 1500)
            sample = {"left_image": new_left_image, "right_image": new_right_image}
        else:
            left_image = sample
            new_left_image = f.crop(left_image, 50, 274, 750, 1500)
            sample = new_left_image
        return sample


class ResizeImage(object):
    def __init__(self, train=True, size=(256, 512)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
        # 训练模式下，样本包含左右图像
        if self.train:
            left_image = sample["left_image"]
            right_image = sample["right_image"]
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            sample = {"left_image": new_left_image, "right_image": new_right_image}
        # 验证或测试模式下，样本为单个图像
        else:
            left_image = sample
            new_left_image = self.transform(left_image)
            sample = new_left_image
        return sample


class DoTest(object):
    def __call__(self, sample):
        # 返回其与水平翻转后的图像的堆叠
        new_sample = torch.stack((sample, torch.flip(sample, [2])))
        return new_sample


class DoVal(object):
    def __call__(self, sample):
        left_image = sample["left_image"]
        right_image = sample["right_image"]
        flipped_left = torch.flip(left_image, [2])
        # 加入水平翻转后的左图
        sample = {
            "left_image": left_image,
            "right_image": right_image,
            "flipped_left_image": flipped_left,
        }
        return sample


class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        if self.train:
            left_image = sample["left_image"]
            right_image = sample["right_image"]
            new_right_image = self.transform(right_image)
            new_left_image = self.transform(left_image)
            sample = {"left_image": new_left_image, "right_image": new_right_image}
        else:
            left_image = sample
            sample = self.transform(left_image)
        return sample


class RandomFlip(object):
    def __init__(self, do_augmentation):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation

    def __call__(self, sample):
        left_image = sample["left_image"]
        right_image = sample["right_image"]
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > 0.5:
                fliped_left = self.transform(right_image)
                fliped_right = self.transform(left_image)
                sample = {"left_image": fliped_left, "right_image": fliped_right}
        else:
            sample = {"left_image": left_image, "right_image": right_image}
        return sample


# 图像对增强类
class AugmentImagePair(object):
    def __init__(self, augment_parameters, do_augmentation):
        self.do_augmentation = do_augmentation
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, sample):
        left_image = sample["left_image"]
        right_image = sample["right_image"]
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if p > 0.5:
                # 随机调整gamma值
                random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                left_image_aug = left_image**random_gamma
                right_image_aug = right_image**random_gamma

                # 随机调整亮度
                random_brightness = np.random.uniform(
                    self.brightness_low, self.brightness_high
                )
                left_image_aug = left_image_aug * random_brightness
                right_image_aug = right_image_aug * random_brightness

                # 随机调整颜色
                random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                for i in range(3):
                    left_image_aug[i, :, :] *= random_colors[i]
                    right_image_aug[i, :, :] *= random_colors[i]

                # 饱和处理
                left_image_aug = torch.clamp(left_image_aug, 0, 1)
                right_image_aug = torch.clamp(right_image_aug, 0, 1)

                sample = {"left_image": left_image_aug, "right_image": right_image_aug}

        else:
            sample = {"left_image": left_image, "right_image": right_image}
        return sample
