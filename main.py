import time
import torch
import os

from utils import *
from options import MainOptions
from data_loader import prepare_dataloader
from architectures import create_architecture


def train(args):
    # 验证函数，评估在验证集上的性能
    def validate(epoch):
        model.to_test()

        # 初始化新的损失项
        model.set_new_loss_item(epoch, train=False)

        # 对于非 WGAN 结构，关闭梯度访问以提高计算效率
        if "wgan" not in args.architecture:
            torch.set_grad_enabled(False)

        # 对于验证集中的每个图像，计算模型的损失
        # i 不可删，因为可能会有-1
        for i, data in enumerate(val_loader):
            model.set_input(data)
            model.forward()
            model.add_running_loss_val(epoch)

        if "wgan" not in args.architecture:
            torch.set_grad_enabled(True)

        # 计算并存储验证集的运行损失
        model.make_running_loss(epoch, val_n_img, train=False)
        return

    # 加载数据
    n_img, loader = prepare_dataloader(args, "train")
    val_n_img, val_loader = prepare_dataloader(args, "val")

    model = create_architecture(args)
    model.set_data_loader(loader)

    if not args.resume:  # 如果不是从之前的检查点恢复训练
        # 初始化最佳验证损失为正无穷
        best_val_loss = float("Inf")

        # 计算预训练的训练损失
        validate(-1)
        # 打印预训练信息
        pre_validation_update(model.losses[-1]["val"])
    else:
        # 从模型的已保存损失中获取最佳验证损失
        best_val_loss = min(
            [model.losses[epoch]["val"]["G"] for epoch in model.losses.keys()]
        )

    # 初始化运行验证损失值
    running_val_loss = 0.0

    # 开始训练
    for epoch in range(model.start_epoch, args.epochs):
        print("Now in epoch " + str(epoch))

        # 更新学习率
        model.update_learning_rate(epoch, args.learning_rate)
        print("learning rate updated")

        c_time = time.time()
        model.to_train()
        model.set_new_loss_item(epoch)  # 设置新的损失项
        print("new loss has been set")

        # 运行一个训练周期
        model.run_epoch(epoch, n_img)

        # 对模型在验证集上进行验证
        validate(epoch)

        # 打印训练和验证损失的更新
        print_epoch_update(epoch, time.time() - c_time, model.losses)

        # 保存检查点，以便可以恢复训练
        running_val_loss = model.losses[epoch]["val"]["G"]  # 更新运行验证损失值
        is_best = running_val_loss < best_val_loss  # 判断当前模型是否为最佳模型
        if is_best:
            best_val_loss = running_val_loss  # 更新最佳验证损失值
        model.save_checkpoint(epoch, is_best, best_val_loss)  # 保存检查点

    print("Finished Training. Best validation loss:\t{:.3f}".format(best_val_loss))

    # 保存最终模型
    model.save_networks("final")
    if running_val_loss != best_val_loss:
        model.save_best_networks()

    model.save_losses()  # 保存损失记录


def test(args):
    # 对于Pilzer，视差已经由它们自己的FuseNet进行了后处理。
    do_post_processing = args.postprocessing and "pilzer" not in args.architecture

    input_height = args.input_height
    input_width = args.input_width

    output_directory = args.output_dir
    n_img, test_loader = prepare_dataloader(args, "test")

    # 创建架构模型
    model = create_architecture(args)
    # 加载训练好的网络
    which_model = "final" if args.load_final else "best"
    model.load_networks(which_model)
    model.to_test()

    # 初始化一个数组来存储视差
    disparities = np.zeros((n_img, input_height, input_width), dtype=np.float32)
    inference_time = 0.0

    # 关闭梯度的计算
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i % 100 == 0 and i != 0:
                print("Testing... Now at image: {}".format(i))

            t_start = time.time()
            # 进行前向传播
            disps = model.fit(data)
            # 这里对输出进行了处理，确保disps是一个二维张量
            # 如果模型输出了多个视差图，则取出第一个元素，然后使用切片取出这个张量的第一个channel的所有内容
            disps = (
                disps[0][:, 0, :, :] if isinstance(disps, tuple) else disps.squeeze()
            )

            # 后处理
            if do_post_processing:
                disparities[i] = post_process_disparity(disps.cpu().numpy())
            else:
                disp = disps.unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
            t_end = time.time()
            inference_time += t_end - t_start

    # 打印时间
    if args.test_time:
        test_time_message = (
            "Inference took {:.4f} seconds. That is {:.2f} imgs/s or {:.6f} s/img."
        )
        print(
            test_time_message.format(
                inference_time, (n_img / inference_time), 1.0 / (n_img / inference_time)
            )
        )

    # 生成保存视差图的文件名
    disp_file_name = "disparities_{}_{}.npy".format(args.dataset, model.name)
    # 拼接保存视差图的完整路径
    full_disp_path = os.path.join(output_directory, disp_file_name)
    # 如果已存在相同文件路径，提示覆盖视差图
    if os.path.exists(full_disp_path):
        print("Overwriting disparities at {}...".format(full_disp_path))
    # 将视差数据保存为.npy文件
    np.save(full_disp_path, disparities)
    print("Finished Testing")


def main():
    parser = MainOptions()
    args = parser.parse()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "verify-data":
        from utils.reduce_image_set import check_if_all_images_are_present

        check_if_all_images_are_present("kitti", args.data_dir)
        check_if_all_images_are_present("eigen", args.data_dir)
        check_if_all_images_are_present("cityscapes", args.data_dir)


if __name__ == "__main__":
    main()

    print("YOU ARE TERMINATED!")
