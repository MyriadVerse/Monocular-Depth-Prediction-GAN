# Import project files
from options import EvaluateOptions
from utils.evaluation_utils import *
from config_parameters import *
import numpy as np
import cv2
import matplotlib

matplotlib.use("TkAgg")  # 使用 TkAgg 后端
import matplotlib.pyplot as plt


def save_depth_rgb_image(depth_map, file_path):
    # 将深度图的值映射到 0 到 1 之间
    normalized_depth_map = (depth_map - np.min(depth_map)) / (
        np.max(depth_map) - np.min(depth_map)
    )

    # 使用伪彩色映射将深度值映射到 RGB 空间
    colormap = plt.cm.jet  # 选择伪彩色映射，可以根据需要选择不同的 colormap
    depth_rgb = (colormap(normalized_depth_map)[:, :, :3] * 255).astype(np.uint8)

    # 保存 RGB 图像
    cv2.imwrite(file_path, cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))


def evaluate_eigen(args, verbose=True):
    # 加载已预测的视差数据和测试数据集
    pred_disparities = np.load(args.predicted_disp_path)
    test_files = sorted(read_text_lines(EIGEN_PATH.format("test")))
    num_samples = len(test_files)

    # 确保已预测视差的图片数量与测试数据集数量相同
    assert_str = "Only {} disparities recovered out of required {}".format(
        len(pred_disparities), num_samples
    )
    assert len(pred_disparities) == num_samples, assert_str

    # gt_files: 存储真实深度图像文件的路径列表
    # gt_calib: 存储相机校准文件的路径列表
    # im_sizes: 存储图像尺寸的列表
    # im_files: 存储输入图像文件的路径列表
    # cams: 存储相机编号的列表
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(
        test_files, args.data_dir
    )

    gt_depths = []
    pred_depths = []

    for t_id in range(num_samples):
        # 获取相机编号，2代表左相机，3代表右相机
        camera_id = cams[t_id]

        # 生成真实深度图
        depth = generate_depth_map(
            gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True
        )
        gt_depths.append(depth.astype(np.float32))

        # 生成预测视差图
        disp_pred = cv2.resize(
            pred_disparities[t_id],
            (im_sizes[t_id][1], im_sizes[t_id][0]),
            interpolation=cv2.INTER_LINEAR,
        )
        disp_pred = disp_pred * disp_pred.shape[1]

        # 生成预测深度图
        focal_length, baseline = get_focal_length_baseline(gt_calib[t_id], camera_id)
        depth_pred = (baseline * focal_length) / disp_pred
        depth_pred[np.isinf(depth_pred)] = 0
        pred_depths.append(depth_pred)

    # 测评指标
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth

        # 生成掩码，过滤深度过大或过小的像素
        mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)

        # 如果启用了Garg或Eigen的裁剪选项
        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape

            # Garg ECCV16
            if args.garg_crop:
                crop = np.array(
                    [
                        0.40810811 * gt_height,
                        0.99189189 * gt_height,
                        0.03594771 * gt_width,
                        0.96405229 * gt_width,
                    ]
                ).astype(np.int32)
            # Eigen NIPS14
            elif args.eigen_crop:
                crop = np.array(
                    [
                        0.3324324 * gt_height,
                        0.91351351 * gt_height,
                        0.0359477 * gt_width,
                        0.96405229 * gt_width,
                    ]
                ).astype(np.int32)

            # 只保留裁剪区域内的有效像素
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0] : crop[1], crop[2] : crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        # 计算性能指标
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(
            gt_depth[mask], pred_depth[mask]
        )

        if i % 100 == 0:
            # 创建输出文件夹
            output_folder = os.path.join("output", "image")
            os.makedirs(output_folder, exist_ok=True)

            # 加载原始图像
            original_image = plt.imread(im_files[i])

            # 创建一个新的图像，将原始图像和预测的深度图像放在同一张图上
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(original_image)
            plt.axis("off")  # 关闭坐标轴

            plt.subplot(2, 1, 2)
            plt.imshow(pred_depth, cmap="jet")  # 可视化深度图
            plt.axis("off")  # 关闭坐标轴

            # 隐藏子图之间的间距和边界
            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)

            # 保存合并后的图像
            plt.savefig(os.path.join(output_folder, f"image_{i}_depth_comparison.png"))
            plt.close()

    if verbose:
        print(
            "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
                "abs_rel", "sq_rel", "rms", "log_rms", "a1", "a2", "a3"
            )
        )
        print(
            "{:10.8f}, {:10.8f}, {:10.8f}, {:10.8f}, {:10.8f}, {:10.8f}, {:10.8f}".format(
                abs_rel.mean(),
                sq_rel.mean(),
                rms.mean(),
                log_rms.mean(),
                a1.mean(),
                a2.mean(),
                a3.mean(),
            )
        )

    return (
        abs_rel.mean(),
        sq_rel.mean(),
        rms.mean(),
        log_rms.mean(),
        a1.mean(),
        a2.mean(),
        a3.mean(),
    )


if __name__ == "__main__":
    parser = EvaluateOptions()
    args = parser.parse()

    if args.split == "eigen":
        evaluate_eigen(args)
    else:
        print("Split not recognised.")
