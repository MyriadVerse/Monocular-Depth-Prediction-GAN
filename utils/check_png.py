import os


def check_png_files(root_dir):
    # 遍历根目录下的所有子目录和文件
    for root, dirs, files in os.walk(root_dir):
        # 检查当前目录中是否有 PNG 文件
        png_files = [file for file in files if file.endswith(".png")]
        if png_files:
            # 如果存在 PNG 文件，则打印出它们所在的目录
            print(f"PNG 文件存在于目录: {root}")
            for png_file in png_files:
                print(f"  - {png_file}")
        # 如果当前目录没有 PNG 文件，则继续遍历下一个目录
    print("over")


# 调用函数并指定根目录
check_png_files("dataset/")
