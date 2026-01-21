# Image Segmentation with segmentation_models_pytorch

This project demonstrates how to use the [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) library for image segmentation with custom DataFrames containing image and mask paths.

## Overview

This implementation provides a complete pipeline for training and evaluating image segmentation models using the segmentation_models_pytorch library. It's designed to work seamlessly with pandas DataFrames containing `images_paths` and `masks_paths` columns.

## Features

- ✅ Custom Dataset class that works with DataFrames
- ✅ Support for multiple architectures (Unet, FPN, PSPNet, DeepLabV3, etc.)
- ✅ 800+ pretrained encoders support
- ✅ Data augmentation using Albumentations
- ✅ Training with metrics (IoU, F-score, Accuracy, etc.)
- ✅ Model evaluation and visualization
- ✅ Binary and multiclass segmentation support

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- segmentation-models-pytorch
- albumentations
- pandas
- opencv-python
- matplotlib

## Dataset Format

Your DataFrames should have the following structure:

```python
import pandas as pd

train_df = pd.DataFrame({
    'images_paths': ['/path/to/image1.jpg', '/path/to/image2.jpg', ...],
    'masks_paths': ['/path/to/mask1.png', '/path/to/mask2.png', ...]
})

valid_df = pd.DataFrame({
    'images_paths': [...],
    'masks_paths': [...]
})

test_df = pd.DataFrame({
    'images_paths': [...],
    'masks_paths': [...]
})
```

## Quick Start

### 1. Using Python Scripts

#### Training

```bash
python train.py \
    --train-csv train.csv \
    --valid-csv valid.csv \
    --encoder resnet34 \
    --architecture Unet \
    --epochs 40 \
    --batch-size 16 \
    --lr 0.0001 \
    --image-size 320 \
    --num-classes 1 \
    --save-dir ./checkpoints \
    --device cuda
```

#### Inference

```bash
python inference.py \
    --test-csv test.csv \
    --model-path ./checkpoints/best_model.pth \
    --encoder resnet34 \
    --batch-size 16 \
    --image-size 320 \
    --num-classes 1 \
    --device cuda \
    --visualize \
    --num-visualize 5 \
    --save-dir ./predictions
```

### 2. Using Python Code

```python
import pandas as pd
from train import train_model
from inference import evaluate_model, visualize_predictions

# Load your DataFrames
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')
test_df = pd.read_csv('test.csv')

# Train the model
model = train_model(
    train_df=train_df,
    valid_df=valid_df,
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
    save_dir='./checkpoints'
)

# Evaluate on test set
import torch
best_model = torch.load('./checkpoints/best_model.pth')

metrics = evaluate_model(
    model=best_model,
    test_df=test_df,
    encoder='resnet34',
    encoder_weights='imagenet',
    device='cuda',
    batch_size=16,
    image_height=320,
    image_width=320,
    num_classes=1,
)

# Visualize predictions
visualize_predictions(
    model=best_model,
    test_df=test_df,
    encoder='resnet34',
    encoder_weights='imagenet',
    device='cuda',
    num_samples=5,
    image_height=320,
    image_width=320,
    save_dir='./predictions'
)
```

### 3. Using Jupyter Notebook

See [example_usage.ipynb](example_usage.ipynb) for a complete walkthrough.

## Supported Architectures

The following architectures are supported:

- **Unet** - [Paper](https://arxiv.org/abs/1505.04597)
- **UnetPlusPlus** - [Paper](https://arxiv.org/pdf/1807.10165.pdf)
- **MAnet** - [Paper](https://ieeexplore.ieee.org/abstract/document/9201310)
- **Linknet** - [Paper](https://arxiv.org/abs/1707.03718)
- **FPN** - [Paper](http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf)
- **PSPNet** - [Paper](https://arxiv.org/abs/1612.01105)
- **PAN** - [Paper](https://arxiv.org/abs/1805.10180)
- **DeepLabV3** - [Paper](https://arxiv.org/abs/1706.05587)
- **DeepLabV3Plus** - [Paper](https://arxiv.org/abs/1802.02611)

## Supported Encoders

800+ encoders are supported, including:

- ResNet family (resnet18, resnet34, resnet50, resnet101, resnet152)
- EfficientNet family (efficientnet-b0 to efficientnet-b7)
- MobileNet family (mobilenet_v2)
- DenseNet family (densenet121, densenet161, densenet169, densenet201)
- VGG family (vgg11, vgg13, vgg16, vgg19)
- And many more from [timm](https://github.com/huggingface/pytorch-image-models)

## Project Structure

```
Image_Segmentation/
├── dataset.py              # Custom Dataset class and augmentation functions
├── train.py                # Training script
├── inference.py            # Inference and evaluation script
├── example_usage.ipynb     # Example Jupyter notebook
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Components

### Dataset Class

The `SegmentationDataset` class in `dataset.py` handles:
- Loading images and masks from DataFrame paths
- Data augmentation using Albumentations
- Preprocessing for different encoders
- Support for both binary and multiclass segmentation

### Training

The training script provides:
- Flexible model configuration
- Multiple loss functions (Dice, Focal, etc.)
- Metrics tracking (IoU, F-score, Accuracy, Recall, Precision)
- Checkpoint saving
- Learning rate scheduling

### Inference

The inference script supports:
- Model evaluation on test data
- Prediction visualization
- Batch processing
- Metrics computation

## Configuration Options

### Model Configuration

```python
ENCODER = 'resnet34'              # Encoder architecture
ENCODER_WEIGHTS = 'imagenet'      # Pretrained weights
ARCHITECTURE = 'Unet'             # Model architecture
ACTIVATION = 'sigmoid'            # 'sigmoid' for binary, 'softmax' for multiclass
NUM_CLASSES = 1                   # Number of output classes
```

### Training Configuration

```python
DEVICE = 'cuda'                   # 'cuda' or 'cpu'
BATCH_SIZE = 16                   # Batch size
NUM_EPOCHS = 40                   # Number of training epochs
LEARNING_RATE = 0.0001           # Learning rate
IMAGE_HEIGHT = 320               # Input image height
IMAGE_WIDTH = 320                # Input image width
```

## Examples

### Binary Segmentation

```python
# For binary segmentation (e.g., foreground/background)
model = train_model(
    train_df=train_df,
    valid_df=valid_df,
    num_classes=1,
    activation='sigmoid',
    # ... other parameters
)
```

### Multiclass Segmentation

```python
# For multiclass segmentation (e.g., 3 classes)
model = train_model(
    train_df=train_df,
    valid_df=valid_df,
    num_classes=3,
    activation='softmax',
    # ... other parameters
)
```

## Tips for Better Performance

1. **Use appropriate encoder**: Start with ResNet34 or EfficientNet-B0 for good speed/accuracy tradeoff
2. **Adjust image size**: Larger images (512x512) generally give better results but require more memory
3. **Data augmentation**: Enable augmentation for better generalization
4. **Batch size**: Use the largest batch size that fits in your GPU memory
5. **Learning rate**: Start with 1e-4 and adjust based on training behavior
6. **Pretrained weights**: Always use pretrained weights (imagenet) for faster convergence

## Metrics

The following metrics are tracked during training and evaluation:

- **IoU (Intersection over Union)**: Measures overlap between prediction and ground truth
- **F-score (Dice coefficient)**: Harmonic mean of precision and recall
- **Accuracy**: Pixel-wise classification accuracy
- **Recall**: True positive rate
- **Precision**: Positive predictive value

## Troubleshooting

### Out of Memory

- Reduce batch size
- Reduce image size
- Use a lighter encoder (e.g., mobilenet_v2)

### Poor Performance

- Increase training epochs
- Add more data augmentation
- Try different encoders/architectures
- Adjust learning rate
- Check data quality and masks

### Slow Training

- Enable mixed precision training
- Use DataLoader with more workers
- Use a lighter architecture

## References

- [segmentation_models_pytorch GitHub](https://github.com/qubvel-org/segmentation_models.pytorch)
- [segmentation_models_pytorch Documentation](https://smp.readthedocs.io/)
- [Binary Segmentation Example](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/main/examples/binary_segmentation_intro.ipynb)

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) by Pavel Yakubovskiy
- [Albumentations](https://github.com/albumentations-team/albumentations) for data augmentation
