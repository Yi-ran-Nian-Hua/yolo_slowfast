#!/bin/bash

# 创建必要的目录
mkdir -p ava_dataset/videos

# 下载训练集和验证集视频列表
echo "下载视频列表..."
curl -L "https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt" -o ava_file_names_trainval_v2.1.txt

# 检查文件是否下载成功
if [ ! -f ava_file_names_trainval_v2.1.txt ]; then
    echo "错误：无法下载视频列表文件"
    exit 1
fi

# 下载视频文件
echo "开始下载视频文件..."
while IFS= read -r filename; do
    if [ -z "$filename" ]; then
        continue
    fi
    
    output_path="ava_dataset/videos/${filename}"
    
    # 如果文件已存在，跳过下载
    if [ -f "$output_path" ]; then
        echo "文件已存在，跳过: $filename"
        continue
    fi
    
    echo "下载: $filename"
    curl -L "https://s3.amazonaws.com/ava-dataset/trainval/${filename}" -o "$output_path"
    
    # 检查下载是否成功
    if [ $? -ne 0 ]; then
        echo "下载失败: $filename"
        # 可以选择是否继续下载其他文件
        # exit 1
    fi
done < ava_file_names_trainval_v2.1.txt

echo "下载完成！"

# 清理临时文件
rm ava_file_names_trainval_v2.1.txt 