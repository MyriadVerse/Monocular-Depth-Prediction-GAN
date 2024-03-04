#!/bin/bash

# $1是第一个参数，表示上级目录
echo $1

# 切换到该目录
cd $1
# 如果文件类型为 Zip，则执行解压和删除操作
for item in *; do
    if [ -n "$(file -b "$item" | grep -o 'Zip')" ]; then
	echo "Now unzipping and removing $item"
        unzip -q "$item" && rm "$item"
    fi
done
