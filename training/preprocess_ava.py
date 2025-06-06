import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def extract_frames(video_path, output_dir, fps=1):
    """从视频中提取帧"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
        
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
            frame_path = os.path.join(output_dir, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            
        frame_count += 1
        
    cap.release()

def process_annotations(annotation_file, output_file):
    """处理标注文件，转换为训练所需的格式"""
    df = pd.read_csv(annotation_file)
    
    # 确保必要的列存在
    required_columns = ['video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"标注文件缺少必要的列: {col}")
    
    # 保存处理后的标注
    df.to_csv(output_file, index=False)
    print(f"处理后的标注已保存到: {output_file}")

def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理训练集
    print("处理训练集...")
    train_output_dir = os.path.join(args.output_dir, 'train')
    os.makedirs(train_output_dir, exist_ok=True)
    
    # 处理验证集
    print("处理验证集...")
    val_output_dir = os.path.join(args.output_dir, 'val')
    os.makedirs(val_output_dir, exist_ok=True)
    
    # 处理标注文件
    print("处理标注文件...")
    process_annotations(
        args.train_annotation,
        os.path.join(args.output_dir, 'train_annotations.csv')
    )
    process_annotations(
        args.val_annotation,
        os.path.join(args.output_dir, 'val_annotations.csv')
    )
    
    # 提取视频帧
    print("提取视频帧...")
    video_dir = args.video_dir
    for video_id in tqdm(os.listdir(video_dir)):
        if not video_id.endswith('.mp4'):
            continue
            
        video_path = os.path.join(video_dir, video_id)
        
        # 确定是训练集还是验证集
        if video_id in pd.read_csv(args.train_annotation)['video_id'].unique():
            output_dir = os.path.join(train_output_dir, video_id.split('.')[0])
        else:
            output_dir = os.path.join(val_output_dir, video_id.split('.')[0])
            
        extract_frames(video_path, output_dir, args.fps)
    
    print("预处理完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True,
                      help='视频文件目录')
    parser.add_argument('--train_annotation', type=str, required=True,
                      help='训练集标注文件路径')
    parser.add_argument('--val_annotation', type=str, required=True,
                      help='验证集标注文件路径')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                      help='处理后的数据保存目录')
    parser.add_argument('--fps', type=int, default=1,
                      help='提取帧的帧率')
    
    args = parser.parse_args()
    main(args) 