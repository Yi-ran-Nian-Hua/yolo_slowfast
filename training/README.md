# AVA数据集训练指南

本文件夹包含用于训练动作识别模型的所有必要文件和脚本。

## 文件说明

- `ava_v2.2/`: AVA数据集标注文件
  - `ava_train_v2.2.csv`: 训练集标注
  - `ava_val_v2.2.csv`: 验证集标注
  - `ava_action_list_v2.2.pbtxt`: 动作类别列表
  - 其他时间戳相关文件

- `download_ava.sh`: 下载AVA数据集标注文件的脚本
- `download_videos.py`: 从YouTube下载视频文件的脚本
- `preprocess_ava.py`: 预处理视频和标注数据的脚本
- `train_ava.py`: 训练模型的脚本

## 使用步骤

1. 下载标注文件：
```bash
./download_ava.sh
```

2. 安装必要的Python包：
```bash
pip install pandas tqdm yt-dlp
```

3. 下载视频文件：
```bash
python download_videos.py
```

4. 预处理数据：
```bash
python preprocess_ava.py
```

5. 开始训练：
```bash
python train_ava.py
```

## 注意事项

1. 确保有足够的磁盘空间（至少100GB）
2. 视频下载可能需要较长时间
3. 建议使用GPU进行训练
4. 如果遇到网络问题，可以考虑使用代理或分批下载视频

## 数据集说明

AVA数据集包含：
- 约430个视频
- 80个动作类别
- 每个视频约15分钟
- 标注包含人物位置框和动作类别

动作类别分为两类：
- PERSON_MOVEMENT：人物动作（如走路、跑步等）
- OBJECT_MANIPULATION：物体操作（如打电话、吃东西等） 