# 训练说明

本目录包含用于训练 YOLO-SlowFast 模型的所有相关代码和脚本。

## 文件说明

- `data_loader.py`: 数据加载器，用于加载和预处理视频数据
- `train_ava.py`: 训练脚本，用于训练模型
- `preprocess_ava.py`: 数据预处理脚本，用于处理 AVA 数据集
- `download_ava_videos.sh`: 下载 AVA 数据集的脚本

## 使用方法

1. 下载数据集：
```bash
./download_ava_videos.sh
```

2. 预处理数据：
```bash
python preprocess_ava.py
```

3. 开始训练：
```bash
python train_ava.py --train_annotation processed_data/train_annotations.csv \
                   --val_annotation processed_data/val_annotations.csv \
                   --video_dir processed_data \
                   --checkpoint_dir checkpoints
```

## 注意事项

1. 确保有足够的磁盘空间（至少 100GB）
2. 建议使用 GPU 进行训练
3. 训练过程可能需要较长时间 