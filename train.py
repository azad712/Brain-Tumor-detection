"""
Brain Tumor Detection - YOLOv8 Training Script
Trains YOLOv8 model for brain tumor detection with bounding boxes
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import torch
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Brain Tumor Detection')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='YOLOv8 model variant (n=nano, s=small, m=medium, l=large, x=xlarge)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (adjust based on GPU memory)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (epochs without improvement)')
    
    # Optimizer parameters
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Optimizer weight decay')
    
    # Data augmentation
    parser.add_argument('--hsv_h', type=float, default=0.015,
                        help='HSV-Hue augmentation')
    parser.add_argument('--hsv_s', type=float, default=0.7,
                        help='HSV-Saturation augmentation')
    parser.add_argument('--hsv_v', type=float, default=0.4,
                        help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0,
                        help='Rotation augmentation (degrees)')
    parser.add_argument('--translate', type=float, default=0.1,
                        help='Translation augmentation')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scaling augmentation')
    parser.add_argument('--shear', type=float, default=0.0,
                        help='Shear augmentation (degrees)')
    parser.add_argument('--perspective', type=float, default=0.0,
                        help='Perspective augmentation')
    parser.add_argument('--flipud', type=float, default=0.0,
                        help='Vertical flip probability')
    parser.add_argument('--fliplr', type=float, default=0.5,
                        help='Horizontal flip probability')
    parser.add_argument('--mosaic', type=float, default=1.0,
                        help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='Mixup augmentation probability')
    
    # Other parameters
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml configuration file')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='brain_tumor_detection',
                        help='Experiment name')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Allow overwriting existing project/name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--optimizer', type=str, default='auto',
                        choices=['SGD', 'Adam', 'AdamW', 'auto'],
                        help='Optimizer type')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (e.g., 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads for data loading')
    parser.add_argument('--cache', type=str, default='',
                        choices=['', 'ram', 'disk'],
                        help='Cache images for faster training')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume training from last checkpoint')
    
    return parser.parse_args()


def check_dataset(data_yaml):
    """Check if dataset configuration is valid"""
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset configuration file not found: {data_yaml}")
    
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key in {data_yaml}: {key}")
    
    # Check if directories exist
    dataset_path = Path(data['path'])
    train_path = dataset_path / data['train']
    val_path = dataset_path / data['val']
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_path}")
    
    print(f"✓ Dataset configuration validated: {data_yaml}")
    print(f"  - Classes: {data['nc']}")
    print(f"  - Train path: {train_path}")
    print(f"  - Val path: {val_path}")
    
    return data


def print_training_info(args, data):
    """Print training configuration"""
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model:              {args.model}")
    print(f"Epochs:             {args.epochs}")
    print(f"Batch size:         {args.batch}")
    print(f"Image size:         {args.imgsz}")
    print(f"Learning rate:      {args.lr0} -> {args.lr0 * args.lrf}")
    print(f"Optimizer:          {args.optimizer}")
    print(f"Device:             {args.device if args.device else 'auto'}")
    print(f"Workers:            {args.workers}")
    print(f"Early stopping:     {args.patience} epochs")
    print(f"Classes:            {data['nc']}")
    print(f"Class names:        {', '.join(data['names'].values())}")
    print("=" * 70 + "\n")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ CUDA not available, training on CPU (will be slow)")
    
    # Validate dataset
    print("\n🔍 Validating dataset...")
    data = check_dataset(args.data)
    
    # Print training info
    print_training_info(args, data)
    
    # Load model
    print(f"📦 Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Resume training if specified
    if args.resume:
        print(f"🔄 Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    
    # Train model
    print("\n🚀 Starting training...\n")
    
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            patience=args.patience,
            
            # Optimizer parameters
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            
            # Data augmentation
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
            
            # Other parameters
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            pretrained=args.pretrained,
            seed=args.seed,
            device=args.device,
            workers=args.workers,
            cache=args.cache,
            
            # Additional settings
            save=True,
            save_period=-1,  # Save checkpoint every epoch (-1 to disable)
            plots=True,
            verbose=True,
        )
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Best model saved to: {model.trainer.best}")
        print(f"Last model saved to: {model.trainer.last}")
        print(f"Results saved to: {model.trainer.save_dir}")
        print("=" * 70)
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print("\n📊 FINAL METRICS:")
            print("-" * 70)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{key:30s}: {value:.4f}")
            print("-" * 70)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

# Made with Bob
