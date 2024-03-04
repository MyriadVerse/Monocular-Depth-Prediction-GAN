# 基于生成对抗网络的单张图像深度信息恢复方法

本仓库实现了基本的训练和测试代码，并应当可以在 `PyTorch2.2.0 + cu118 + python3.9` 的环境下运行，可能还需要 `scipy`, `torchvision`, `numpy`, `opencv_python`, `Pillow`，`matplotlib` 等
## 训练

可以通过指定数据集目录、模型名称、架构、网络等参数来训练模型：

```shell

python main.py --data_dir dataset/ --model_name [MODEL_NAME] --architecture gan_single

```

通过在命令行中添加恢复标志并附上保存模型的路径可以恢复训练：

```shell

python main.py --data_dir dataset/ --model_name [MODEL_NAME] --architecture gan_single --resume saved_models/[MODEL_NAME]/model_best.pth.tar

```

训练模型的选项有很多很多，可参考 [options](options/) 文件夹，其中包含三个 python 文件，其中包含用于训练、测试和评估的选项。
## 测试

若要测试，请将上文的“--mode”标志更改为“test”，网络将在 output 文件夹中输出视差

```shell

python main.py --data_dir dataset/ --model_name [MODEL_NAME] --mode test

```

## 深度估计

假设 output 文件夹中存在视差文件，请运行以下脚本来运行评估：

```shell

python evaluate.py --data_dir dataset/ --predicted_disp_path output/disparities_[DATASET]_[MODEL_NAME].npy  

```

## 数据

使用了两个数据集：KITTI 和 CityScapes，由于初始数据集是 png 格式，故训练前可使用 [png2jpg](utils/png2jpg.py) 脚本转换为 jpg 格式

### [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php)

在这项工作中，**eigen** 的分割用于训练和测试模型。该集包含 22600 个训练图像、888 个验证图像和 697 个测试图像

在 [filenames](utils/filenames) 文件夹中，有一些列表详细说明了哪些图像对应于哪个集合。所有数据都可以通过运行下面的脚本下载（中国大陆可能连接不稳定，如果 https 链接无响应，可以尝试使用 http 链接）：

```shell

wget -i utils/kitti_archives_to_download.txt -P ~/my/output/folder/

```

### [CityScapes](https://www.cityscapes-dataset.com)

要访问 CityScapes 数据集的数据，必须注册一个帐户，然后请求对 ground truth 视差的特殊访问权限

使用此数据时，应将以下目录放入 [dataset](dataset/) 文件夹中：
cs_camera/ 包含所有相机参数
cs_disparity/ 具有所有真实视差
cs_leftImg8bit/ 包含所有左侧图像
cs_rightImg8bit/ 包含所有右侧图像

## 参考文献

1. Aleotti F, Tosi F, Poggi M, et al. Generative adversarial networks for unsupervised monocular depth prediction[C]//Proceedings of the European conference on computer vision (ECCV) workshops. 2018: 0-0.
2. Groenendijk R, Karaoglu S, Gevers T, et al. On the benefit of adversarial training for monocular depth estimation[J]. Computer Vision and Image Understanding, 2020, 190: 102848.