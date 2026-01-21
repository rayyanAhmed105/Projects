"""
Quick start example for Image Segmentation with segmentation_models_pytorch.

This script shows the minimal code needed to use the segmentation pipeline.
"""

import pandas as pd
import torch
from train import train_model
from inference import evaluate_model

# Example 1: Create sample DataFrames
# Replace these with your actual data paths
train_df = pd.DataFrame({
    'images_paths': [
        '/path/to/train/image1.jpg',
        '/path/to/train/image2.jpg',
        # Add more training images...
    ],
    'masks_paths': [
        '/path/to/train/mask1.png',
        '/path/to/train/mask2.png',
        # Add more training masks...
    ]
})

valid_df = pd.DataFrame({
    'images_paths': [
        '/path/to/valid/image1.jpg',
        '/path/to/valid/image2.jpg',
        # Add more validation images...
    ],
    'masks_paths': [
        '/path/to/valid/mask1.png',
        '/path/to/valid/mask2.png',
        # Add more validation masks...
    ]
})

test_df = pd.DataFrame({
    'images_paths': [
        '/path/to/test/image1.jpg',
        '/path/to/test/image2.jpg',
        # Add more test images...
    ],
    'masks_paths': [
        '/path/to/test/mask1.png',
        '/path/to/test/mask2.png',
        # Add more test masks...
    ]
})

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(valid_df)}")
print(f"Test samples: {len(test_df)}")

# Example 2: Train a model
if __name__ == '__main__':
    # Uncomment and run to train
    """
    model = train_model(
        train_df=train_df,
        valid_df=valid_df,
        encoder='resnet34',
        encoder_weights='imagenet',
        architecture='Unet',
        activation='sigmoid',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=16,
        num_epochs=40,
        learning_rate=0.0001,
        image_height=320,
        image_width=320,
        num_classes=1,  # 1 for binary segmentation
        save_dir='./checkpoints'
    )
    """
    
    # Example 3: Evaluate a trained model
    """
    best_model = torch.load('./checkpoints/best_model.pth')
    
    metrics = evaluate_model(
        model=best_model,
        test_df=test_df,
        encoder='resnet34',
        encoder_weights='imagenet',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=16,
        image_height=320,
        image_width=320,
        num_classes=1,
    )
    
    print("Test Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    """
    
    print("\nQuick Start Instructions:")
    print("1. Update the DataFrame paths above with your actual image and mask paths")
    print("2. Uncomment the training code block and run to train a model")
    print("3. Uncomment the evaluation code block to evaluate the trained model")
    print("4. See README.md for more detailed usage examples")
