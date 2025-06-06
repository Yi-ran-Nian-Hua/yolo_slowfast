import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorchvideo.models.hub import slowfast_r50_detection
from data_loader import AVADataset
import argparse
from tqdm import tqdm
import os

def train(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_dataset = AVADataset(
        annotation_file=args.train_annotation,
        video_dir=args.video_dir,
        num_frames=args.num_frames,
        crop_size=args.crop_size
    )
    
    val_dataset = AVADataset(
        annotation_file=args.val_annotation,
        video_dir=args.video_dir,
        num_frames=args.num_frames,
        crop_size=args.crop_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 加载模型
    model = slowfast_r50_detection(True)
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
        for batch in pbar:
            frames = batch['frames'].to(device)
            boxes = batch['boxes'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(frames, boxes)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': train_loss / (pbar.n + 1),
                'acc': 100. * train_correct / train_total
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
            for batch in pbar:
                frames = batch['frames'].to(device)
                boxes = batch['boxes'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(frames, boxes)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': val_loss / (pbar.n + 1),
                    'acc': 100. * val_correct / val_total
                })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(args.checkpoint_dir, 'best_model.pth'))
            
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100.*train_correct/train_total:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100.*val_correct/val_total:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_annotation', type=str, required=True,
                      help='训练集标注文件路径')
    parser.add_argument('--val_annotation', type=str, required=True,
                      help='验证集标注文件路径')
    parser.add_argument('--video_dir', type=str, required=True,
                      help='视频文件目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='模型检查点保存目录')
    parser.add_argument('--num_frames', type=int, default=32,
                      help='每个视频采样的帧数')
    parser.add_argument('--crop_size', type=int, default=224,
                      help='视频裁剪大小')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='数据加载器的工作进程数')
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                      help='训练设备')
    
    args = parser.parse_args()
    train(args) 