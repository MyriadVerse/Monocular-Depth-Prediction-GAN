import numpy as np
import os
import cv2
from collections import Counter
from scipy.interpolate import LinearNDInterpolator
from numpy import inf


# 计算误差指标
def compute_errors(gt, pred):
    # 真实深度和预测深度之间的比值
    thresh = np.maximum((gt / pred), (pred / gt))
    # 分别表示比值小于1.25、1.25^2、1.25^3的比率
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()
    # 平方根误差（Root Mean Square Error）
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    # 对数平方根误差
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    # 绝对相对误差
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    # 平方相对误差
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# ---- EIGEN ----
def read_text_lines(file_path):
    f = open(file_path, "r")
    lines = f.readlines()
    f.close()
    lines = [l.rstrip() for l in lines]
    return lines


# 从文件列表中读取数据
def read_file_data(files, data_root):
    gt_files = []
    gt_calib = []
    im_sizes = []
    im_files = []
    cams = []

    for filename in files:
        filename = filename.split()[0]
        splits = filename.split("/")
        date = splits[0]
        im_id = splits[4][:10]

        # 构建图像路径和velodyne文件路径
        image_path = os.path.join(data_root, filename)
        velodyne_name = "{}/{}/velodyne_points/data/{}.bin".format(
            splits[0], splits[1], im_id
        )

        if os.path.isfile(image_path):
            gt_files.append(os.path.join(data_root, velodyne_name))
            gt_calib.append(data_root + date + "/")
            im_sizes.append(cv2.imread(image_path).shape[:2])
            im_files.append(image_path)
            cams.append(2)

    return gt_files, gt_calib, im_sizes, im_files, cams


# 从Velodyne数据文件中加载点云数据
def load_velodyne_points(file_name):
    # https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


# 对深度图像的线性插值
def lin_interp(shape, xyd):
    # https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


# 解析标定文件
def read_calib_file(path):
    # https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            value = value.strip()
            data[key] = value
            # 检查值中是否包含浮点数字符
            if float_chars.issuperset(value):
                # 尝试转换为浮点数组
                try:
                    data[key] = np.array(list(map(float, value.split(" "))))
                except ValueError:
                    pass

    return data


# 从标定文件中获取相机的焦距和基线
def get_focal_length_baseline(calib_dir, cam):
    cam2cam = read_calib_file(calib_dir + "calib_cam_to_cam.txt")
    # 获取左右相机的投影矩阵
    P2_rect = cam2cam["P_rect_02"].reshape(3, 4)
    P3_rect = cam2cam["P_rect_03"].reshape(3, 4)

    # 摄像头 2 位于摄像头 0 左侧 -6cm
    # 摄像头 3 位于摄像头 0 右侧 +54cm
    b2 = P2_rect[0, 3] / -P2_rect[0, 0]
    b3 = P3_rect[0, 3] / -P3_rect[0, 0]
    baseline = b3 - b2

    # 根据相机索引获取相应相机的焦距
    if cam == 2:
        focal_length = P2_rect[0, 0]
    elif cam == 3:
        focal_length = P3_rect[0, 0]
    return focal_length, baseline


# 将二维矩阵中的行列坐标转换为对应的一维索引
def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(
    calib_dir, velo_file_name, im_shape, cam=2, interp=False, vel_depth=False
):
    # 加载相机校准文件
    cam2cam = read_calib_file(calib_dir + "calib_cam_to_cam.txt")
    velo2cam = read_calib_file(calib_dir + "calib_velo_to_cam.txt")
    velo2cam = np.hstack((velo2cam["R"].reshape(3, 3), velo2cam["T"][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # 计算投影矩阵 velodyne->图像平面
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam["R_rect_00"].reshape(3, 3)
    P_rect = cam2cam["P_rect_0" + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # 加载velodyne点云并移除所有在图像平面后方的点（近似）
    # velodyne数据的每一行是：前方距离、左方距离、上方距离、反射率
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # 将点云投影到相机平面上
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # 检查是否在范围内
    # 使用减 1 得到与 KITTI matlab 代码完全相同的值
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = (
        val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    )
    velo_pts_im = velo_pts_im[val_inds, :]

    # 投影到图像上
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = (
        velo_pts_im[:, 2]
    )

    # 查找重叠点并选择最近的深度
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    if interp:
        # 对深度图进行插值以填充空洞
        depth_interp = lin_interp(im_shape, velo_pts_im)
        return depth, depth_interp
    return depth
