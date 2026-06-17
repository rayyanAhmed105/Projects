"""
Training script for Image Segmentation using segmentation_models_pytorch.

This script demonstrates how to train a segmentation model using
segmentation_models_pytorch library with custom DataFrames containing
'images_paths' and 'masks_paths' columns.
"""

import os
import torch
import pandas as pd
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import (
    SegmentationDataset,
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing
)


class Trainer:
    """
    Training class for segmentation models.
    """
    
    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        device='cpu',
        verbose=True
    ):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose
        
    def train_epoch(self, train_loader):
        """Run one training epoch."""
        self.model.train()
        
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        
        with tqdm(train_loader, desc="Training", unit="batch", disable=not self.verbose) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                prediction = self.model(x)
                loss = self.loss(prediction, y)
                loss.backward()
                self.optimizer.step()
                
                # Update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)
                
                # Update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(prediction, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
                
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        
        return logs
    
    def valid_epoch(self, valid_loader):
        """Run one validation epoch."""
        self.model.eval()
        
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        
        with tqdm(valid_loader, desc="Validation", unit="batch", disable=not self.verbose) as iterator:
            for x, y in iterator:
                with torch.no_grad():
                    x, y = x.to(self.device), y.to(self.device)
                    prediction = self.model(x)
                    loss = self.loss(prediction, y)
                
                # Update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)
                
                # Update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(prediction, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)
                
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
        
        return logs
    
    @staticmethod
    def _format_logs(logs):
        """Format logs for display."""
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        return ', '.join(str_logs)


class AverageValueMeter:
    """Compute and store the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def add(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
    
    @property
    def mean(self):
        return self.avg


def train_model(
    train_df,
    valid_df,
    encoder='resnet34',
    encoder_weights='imagenet',
    architecture='Unet',
    activation='sigmoid',
    device='cuda',
    batch_size=16,
    num_epochs=40,
    learning_rate=0.0001,
    image_height=320,
    image_width=320,
    num_classes=1,
    class_rgb_values=None,
    save_dir='./checkpoints'
):
    """
    Train a segmentation model.
    
    Args:
        train_df (pd.DataFrame): Training DataFrame with 'images_paths' and 'masks_paths' columns
        valid_df (pd.DataFrame): Validation DataFrame with 'images_paths' and 'masks_paths' columns
        encoder (str): Encoder name (e.g., 'resnet34', 'efficientnet-b0')
        encoder_weights (str): Encoder weights (e.g., 'imagenet')
        architecture (str): Model architecture (e.g., 'Unet', 'FPN', 'PSPNet')
        activation (str): Activation function ('sigmoid' for binary, 'softmax' for multiclass)
        device (str): Device to use ('cuda' or 'cpu')
        batch_size (int): Batch size
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        image_height (int): Image height
        image_width (int): Image width
        num_classes (int): Number of classes
        class_rgb_values (list): RGB values for each class (for multiclass)
        save_dir (str): Directory to save checkpoints
    """
    
    # Create model
    model_class = getattr(smp, architecture)
    model = model_class(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=activation,
    )
    
    # Get preprocessing function for encoder
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    
    # Create datasets
    train_dataset = SegmentationDataset(
        train_df,
        augmentation=get_training_augmentation(image_height, image_width),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_rgb_values,
    )
    
    valid_dataset = SegmentationDataset(
        valid_df,
        augmentation=get_validation_augmentation(image_height, image_width),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_rgb_values,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Define loss function
    if num_classes == 1:
        # Binary segmentation
        loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
    else:
        # Multiclass segmentation
        loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=False)
    
    # Define metrics
    metrics = [
        smp.metrics.IoU(threshold=0.5),
        smp.metrics.Fscore(threshold=0.5),
    ]
    
    # Define optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=learning_rate),
    ])
    
    # Move model to device
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
    
    # Create trainer
    trainer = Trainer(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True
    )
    
    # Create checkpoint directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    max_score = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch + 1}/{num_epochs}')
        
        train_logs = trainer.train_epoch(train_loader)
        valid_logs = trainer.valid_epoch(valid_loader)
        
        # Save model if validation score improved
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, os.path.join(save_dir, 'best_model.pth'))
            print(f'Model saved! IoU score improved to {max_score:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_logs': train_logs,
                'valid_logs': valid_logs,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    print(f'\nTraining completed! Best IoU score: {max_score:.4f}')
    
    return model


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--train-csv', type=str, required=True, help='Path to training CSV file')
    parser.add_argument('--valid-csv', type=str, required=True, help='Path to validation CSV file')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder name')
    parser.add_argument('--architecture', type=str, default='Unet', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=320, help='Image size (height and width)')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load DataFrames
    train_df = pd.read_csv(args.train_csv)
    valid_df = pd.read_csv(args.valid_csv)
    
    # Train model
    train_model(
        train_df=train_df,
        valid_df=valid_df,
        encoder=args.encoder,
        architecture=args.architecture,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_height=args.image_size,
        image_width=args.image_size,
        num_classes=args.num_classes,
        save_dir=args.save_dir,
        device=args.device,
    )
