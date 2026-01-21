"""
Inference script for Image Segmentation using trained models.

This script demonstrates how to use a trained segmentation model
for inference on test data.
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from dataset import (
    SegmentationDataset,
    get_validation_augmentation,
    get_preprocessing
)


def visualize_predictions(
    model,
    test_df,
    encoder='resnet34',
    encoder_weights='imagenet',
    device='cuda',
    num_samples=5,
    image_height=320,
    image_width=320,
    class_rgb_values=None,
    save_dir='./predictions'
):
    """
    Visualize model predictions on test data.
    
    Args:
        model: Trained segmentation model
        test_df (pd.DataFrame): Test DataFrame with 'images_paths' and 'masks_paths' columns
        encoder (str): Encoder name
        encoder_weights (str): Encoder weights
        device (str): Device to use
        num_samples (int): Number of samples to visualize
        image_height (int): Image height
        image_width (int): Image width
        class_rgb_values (list): RGB values for each class
        save_dir (str): Directory to save predictions
    """
    import cv2
    
    # Get preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    
    # Create test dataset
    test_dataset = SegmentationDataset(
        test_df,
        augmentation=get_validation_augmentation(image_height, image_width),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_rgb_values,
    )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
    
    # Visualize predictions
    for i in range(min(num_samples, len(test_dataset))):
        image, gt_mask = test_dataset[i]
        
        # Add batch dimension
        x_tensor = torch.from_numpy(image).unsqueeze(0)
        if torch.cuda.is_available() and device == 'cuda':
            x_tensor = x_tensor.cuda()
        
        # Predict
        with torch.no_grad():
            pred_mask = model(x_tensor)
            pred_mask = pred_mask.squeeze().cpu().numpy()
        
        # Convert tensors to numpy for visualization
        image_vis = image.transpose(1, 2, 0)  # CHW -> HWC
        
        # Denormalize image for visualization if needed
        image_vis = (image_vis - image_vis.min()) / (image_vis.max() - image_vis.min())
        
        # Prepare ground truth mask for visualization
        if len(gt_mask.shape) == 3:
            gt_mask_vis = gt_mask.transpose(1, 2, 0)
        else:
            gt_mask_vis = gt_mask
        
        # Prepare prediction mask
        if len(pred_mask.shape) == 2:
            # Binary segmentation
            pred_mask_vis = (pred_mask > 0.5).astype(np.uint8)
        else:
            # Multiclass segmentation
            pred_mask_vis = np.argmax(pred_mask, axis=0)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_vis)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        if len(gt_mask_vis.shape) == 3 and gt_mask_vis.shape[-1] > 1:
            axes[1].imshow(np.argmax(gt_mask_vis, axis=-1))
        else:
            axes[1].imshow(gt_mask_vis, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask_vis, cmap='gray' if len(pred_mask.shape) == 2 else None)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'prediction_{i}.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f'Saved prediction {i} to {save_path}')


def evaluate_model(
    model,
    test_df,
    encoder='resnet34',
    encoder_weights='imagenet',
    device='cuda',
    batch_size=16,
    image_height=320,
    image_width=320,
    num_classes=1,
    class_rgb_values=None,
):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained segmentation model
        test_df (pd.DataFrame): Test DataFrame with 'images_paths' and 'masks_paths' columns
        encoder (str): Encoder name
        encoder_weights (str): Encoder weights
        device (str): Device to use
        batch_size (int): Batch size
        image_height (int): Image height
        image_width (int): Image width
        num_classes (int): Number of classes
        class_rgb_values (list): RGB values for each class
        
    Returns:
        dict: Evaluation metrics
    """
    from tqdm import tqdm
    
    # Get preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    
    # Create test dataset
    test_dataset = SegmentationDataset(
        test_df,
        augmentation=get_validation_augmentation(image_height, image_width),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_rgb_values,
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Define metrics
    metrics = [
        smp.metrics.IoU(threshold=0.5),
        smp.metrics.Fscore(threshold=0.5),
        smp.metrics.Accuracy(threshold=0.5),
        smp.metrics.Recall(threshold=0.5),
        smp.metrics.Precision(threshold=0.5),
    ]
    
    # Set model to evaluation mode
    model.eval()
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
    
    # Evaluate
    metrics_values = {metric.__name__: [] for metric in metrics}
    
    with tqdm(test_loader, desc="Evaluating", unit="batch") as iterator:
        for x, y in iterator:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                prediction = model(x)
                
                # Calculate metrics
                for metric_fn in metrics:
                    metric_value = metric_fn(prediction, y).cpu().numpy()
                    metrics_values[metric_fn.__name__].append(metric_value)
    
    # Calculate mean metrics
    metrics_mean = {k: np.mean(v) for k, v in metrics_values.items()}
    
    # Print results
    print('\nEvaluation Results:')
    print('-' * 40)
    for metric_name, metric_value in metrics_mean.items():
        print(f'{metric_name}: {metric_value:.4f}')
    print('-' * 40)
    
    return metrics_mean


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    parser.add_argument('--test-csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--encoder', type=str, default='resnet34', help='Encoder name')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--image-size', type=int, default=320, help='Image size (height and width)')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--num-visualize', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--save-dir', type=str, default='./predictions', help='Directory to save predictions')
    
    args = parser.parse_args()
    
    # Load test DataFrame
    test_df = pd.read_csv(args.test_csv)
    
    # Load model
    model = torch.load(args.model_path)
    
    # Evaluate model
    evaluate_model(
        model=model,
        test_df=test_df,
        encoder=args.encoder,
        device=args.device,
        batch_size=args.batch_size,
        image_height=args.image_size,
        image_width=args.image_size,
        num_classes=args.num_classes,
    )
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(
            model=model,
            test_df=test_df,
            encoder=args.encoder,
            device=args.device,
            num_samples=args.num_visualize,
            image_height=args.image_size,
            image_width=args.image_size,
            save_dir=args.save_dir,
        )
