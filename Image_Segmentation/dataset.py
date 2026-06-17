"""
Custom Dataset for Image Segmentation with DataFrame support.

This dataset class is designed to work with pandas DataFrames containing
'images_paths' and 'masks_paths' columns.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Custom Dataset for image segmentation that works with DataFrames.
    
    Args:
        df (pd.DataFrame): DataFrame with 'images_paths' and 'masks_paths' columns
        augmentation (albumentations.Compose, optional): Data augmentation pipeline
        preprocessing (albumentations.Compose, optional): Data preprocessing pipeline
        class_rgb_values (list, optional): RGB values for each class in multiclass segmentation
    """
    
    def __init__(
        self, 
        df,
        augmentation=None, 
        preprocessing=None,
        class_rgb_values=None,
    ):
        self.df = df
        self.images_fps = df['images_paths'].tolist()
        self.masks_fps = df['masks_paths'].tolist()
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.class_rgb_values = class_rgb_values
    
    def __getitem__(self, i):
        
        # Read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_UNCHANGED)
        
        # Handle different mask formats
        if len(mask.shape) == 3:
            # If mask is RGB, convert to grayscale or class indices
            if self.class_rgb_values is not None:
                # For multiclass segmentation with RGB masks
                mask = self.one_hot_encode(mask, self.class_rgb_values)
            else:
                # Convert to grayscale
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
    
    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def one_hot_encode(label, label_values):
        """
        Convert RGB mask to class indices.
        
        Args:
            label: RGB mask
            label_values: List of RGB values for each class
            
        Returns:
            Mask with class indices
        """
        semantic_map = []
        for color in label_values:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
        return semantic_map


def to_tensor(x, **kwargs):
    """Convert image or mask to PyTorch tensor."""
    return x.transpose(2, 0, 1).astype('float32')


def get_training_augmentation(height=320, width=320):
    """
    Get training augmentation pipeline.
    
    Args:
        height: Target height for images
        width: Target width for images
        
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    import albumentations as A
    
    train_transform = [
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        A.GridDistortion(p=0.5),
        A.Perspective(p=0.5),
        A.OneOf([
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ], p=0.9),
        A.OneOf([
            A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.9),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.9),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(height=320, width=320):
    """
    Get validation augmentation pipeline (only resizing).
    
    Args:
        height: Target height for images
        width: Target width for images
        
    Returns:
        albumentations.Compose: Augmentation pipeline
    """
    import albumentations as A
    
    test_transform = [
        A.Resize(height, width),
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """
    Get preprocessing pipeline for encoder.
    
    Args:
        preprocessing_fn: Preprocessing function from encoder
        
    Returns:
        albumentations.Compose: Preprocessing pipeline
    """
    import albumentations as A
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)
