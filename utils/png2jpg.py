import os
from PIL import Image
import time

# 原始PNG图片存储路径
png_root = "dataset/"

# 遍历所有子文件夹和文件
for root, dirs, files in os.walk(png_root):
    for file in files:
        # 检查文件是否以.png结尾
        if file.endswith(".png"):
            # 构建PNG文件的完整路径
            png_path = os.path.join(root, file)
            # 打开PNG图片
            with Image.open(png_path) as img:
                # 构建JPG文件的路径，将原始文件的扩展名替换为.jpg
                jpg_path = os.path.splitext(png_path)[0] + ".jpg"
                # # 将PNG图片保存为JPG格式
                img.convert("RGB").save(jpg_path)
                os.remove(png_path)
