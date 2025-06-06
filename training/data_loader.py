import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize

class AVADataset(Dataset):
    def __init__(self, annotation_file, video_dir, num_frames=32, crop_size=224):
        self.annotations = pd.read_csv(annotation_file)
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.crop_size = crop_size
        
        # 获取唯一的视频ID
        self.video_ids = self.annotations['video_id'].unique()
        
    def __len__(self):
        return len(self.video_ids)
    
    def load_video(self, video_id):
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        return np.array(frames)
    
    def get_annotations(self, video_id):
        video_anns = self.annotations[self.annotations['video_id'] == video_id]
        return video_anns
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # 加载视频帧
        frames = self.load_video(video_id)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2)  # [C, T, H, W]
        
        # 获取该视频的标注
        video_anns = self.get_annotations(video_id)
        
        # 预处理视频帧
        frames = uniform_temporal_subsample(frames, self.num_frames)
        frames = frames.float() / 255.0
        
        # 获取边界框
        boxes = video_anns[['x1', 'y1', 'x2', 'y2']].values
        boxes = torch.from_numpy(boxes)
        
        # 缩放视频和边界框
        frames, boxes = short_side_scale_with_boxes(
            frames, size=self.crop_size, boxes=boxes
        )
        
        # 标准化
        frames = normalize(
            frames,
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225]
        )
        
        # 获取动作标签
        labels = video_anns['action_id'].values
        labels = torch.from_numpy(labels)
        
        return {
            'frames': frames,
            'boxes': boxes,
            'labels': labels,
            'video_id': video_id
        } 