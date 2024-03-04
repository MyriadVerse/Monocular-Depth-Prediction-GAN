import numpy as np


# 后处理立体视差图
def post_process_disparity(disp):
    (_, h, w) = disp.shape
    # 计算平均视差图
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)

    # 创建规范化坐标的网格
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # 计算左右掩码
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    # 使用掩码将左右视差图组合起来生成最终的视差图
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


# 打印每个epoch的更新信息
def print_epoch_update(epoch, time, losses):
    train_losses = losses[epoch]["train"]
    val_losses = losses[epoch]["val"]

    train_loss_string = "train: \t["
    val_loss_string = "val:   \t["
    for key in train_losses.keys():
        train_loss_string += "  {}: {:.4f}  ".format(key, train_losses[key])
        val_loss_string += "  {}: {:.4f}  ".format(key, val_losses[key])

    update_print_statement = (
        "Epoch: {}\t | train: {:.2f}\t | val: {:.2f}\t | time: {:.2f}\n  {}]\n  {}]"
    )
    print(
        update_print_statement.format(
            epoch,
            losses[epoch]["train"]["G"],
            losses[epoch]["val"]["G"],
            time,
            train_loss_string,
            val_loss_string,
        )
    )
    return


# 打印预训练验证集损失更新信息
def pre_validation_update(val_losses):
    val_loss_string = "Pre-training val losses:\t["
    for key in val_losses.keys():
        val_loss_string += "  {}: {:.4f}  ".format(key, val_losses[key])
    val_loss_string += "]"
    print(val_loss_string)
    return
