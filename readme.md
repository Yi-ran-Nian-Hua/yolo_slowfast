# YOLO-SlowFast 动作识别系统

这是一个结合了YOLOv5目标检测和SlowFast网络动作识别的系统，可以实时检测视频中的人物并进行动作识别。

## 功能特点

- 使用YOLOv5进行人物检测
- 使用DeepSORT进行目标跟踪
- 使用SlowFast网络进行动作识别
- 支持实时视频处理
- 支持多种动作类别识别

## 项目结构

```
.
├── yolo_slowfast.py      # 主程序
├── requirements.txt      # 依赖包列表
├── training/            # 训练相关代码
│   ├── README.md        # 训练说明
│   ├── data_loader.py   # 数据加载器
│   ├── train_ava.py     # 训练脚本
│   ├── preprocess_ava.py # 数据预处理
│   └── download_ava_videos.sh # 数据下载脚本
├── deep_sort/           # DeepSORT跟踪算法
├── selfutils/           # 工具函数
└── demo/               # 示例视频
```

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/Yi-ran-Nian-Hua/yolo_slowfast.git
cd yolo_slowfast
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载预训练模型：
```bash
# YOLOv5模型会自动下载
# SlowFast模型会在运行时自动下载
```

## 使用方法

1. 运行实时视频处理：
```bash
python yolo_slowfast.py --input 0  # 使用摄像头
# 或
python yolo_slowfast.py --input video.mp4  # 处理视频文件
```

2. 参数说明：
- `--input`: 输入源（摄像头ID或视频文件路径）
- `--output`: 输出视频保存路径
- `--conf`: 目标检测置信度阈值
- `--device`: 运行设备（cpu/cuda）
- `--show`: 是否显示处理过程

## 训练自己的模型

1. 准备数据：
```bash
cd training
./download_ava_videos.sh  # 下载AVA数据集
python preprocess_ava.py  # 预处理数据
```

2. 开始训练：
```bash
python train_ava.py --train_annotation processed_data/train_annotations.csv \
                   --val_annotation processed_data/val_annotations.csv \
                   --video_dir processed_data \
                   --checkpoint_dir checkpoints
```

## 注意事项

1. 确保有足够的GPU内存（建议至少8GB）
2. 训练过程需要大量磁盘空间（至少100GB）
3. 视频处理可能需要较长时间

## 许可证

MIT License

## 致谢

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [SlowFast Networks](https://github.com/facebookresearch/SlowFast)
- [DeepSORT](https://github.com/nwojke/deep_sort)
- [AVA Dataset](https://research.google.com/ava/)

### Stargazers over time

[![Stargazers over time](https://starchart.cc/wufan-tb/yolo_slowfast.svg)](https://starchart.cc/wufan-tb/yolo_slowfast)


