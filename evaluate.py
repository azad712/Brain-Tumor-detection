"""
Brain Tumor Detection - YOLOv8 Evaluation Script
Evaluates trained model on validation set with comprehensive metrics
"""

import argparse
import os
from pathlib import Path
import yaml
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 Brain Tumor Detection Model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--data', type=str, default='data.yaml',
                        help='Path to data.yaml configuration file')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (e.g., 0 or cpu)')
    parser.add_argument('--save_json', action='store_true',
                        help='Save results to JSON file')
    parser.add_argument('--save_hybrid', action='store_true',
                        help='Save hybrid version of labels')
    parser.add_argument('--project', type=str, default='runs/evaluate',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Allow overwriting existing project/name')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='Generate evaluation plots')
    
    return parser.parse_args()


def print_evaluation_header(args):
    """Print evaluation configuration"""
    print("\n" + "=" * 70)
    print("BRAIN TUMOR DETECTION - MODEL EVALUATION")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Data:        {args.data}")
    print(f"Split:       {args.split}")
    print(f"Batch size:  {args.batch}")
    print(f"Image size:  {args.imgsz}")
    print(f"Confidence:  {args.conf}")
    print(f"IoU:         {args.iou}")
    print(f"Device:      {args.device if args.device else 'auto'}")
    print("=" * 70 + "\n")


def print_metrics_summary(metrics, class_names):
    """Print comprehensive metrics summary"""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 70)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS:")
    print("-" * 70)
    
    metric_names = {
        'metrics/precision(B)': 'Precision (Box)',
        'metrics/recall(B)': 'Recall (Box)',
        'metrics/mAP50(B)': 'mAP@0.5',
        'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
    }
    
    for key, name in metric_names.items():
        if key in metrics:
            value = metrics[key]
            print(f"{name:<25}: {value:.4f}")
    
    # Per-class metrics
    if class_names:
        print("\n📈 PER-CLASS METRICS:")
        print("-" * 70)
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}")
        print("-" * 70)
        
        for cls_id, cls_name in class_names.items():
            precision_key = f'metrics/precision(B)_{cls_id}'
            recall_key = f'metrics/recall(B)_{cls_id}'
            map50_key = f'metrics/mAP50(B)_{cls_id}'
            map50_95_key = f'metrics/mAP50-95(B)_{cls_id}'
            
            precision = metrics.get(precision_key, 0.0)
            recall = metrics.get(recall_key, 0.0)
            map50 = metrics.get(map50_key, 0.0)
            map50_95 = metrics.get(map50_95_key, 0.0)
            
            print(f"{cls_name:<20} {precision:<12.4f} {recall:<12.4f} {map50:<12.4f} {map50_95:<12.4f}")
    
    print("=" * 70)


def print_confusion_matrix_summary(metrics):
    """Print confusion matrix summary if available"""
    if 'confusion_matrix' in metrics:
        print("\n" + "=" * 70)
        print("CONFUSION MATRIX")
        print("=" * 70)
        print("Confusion matrix plot saved in results directory")
        print("=" * 70)


def save_metrics_to_json(metrics, save_path):
    """Save metrics to JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            json_metrics[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            json_metrics[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            json_metrics[key] = int(value)
        else:
            json_metrics[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(json_metrics, f, indent=4)
    
    print(f"\n✓ Metrics saved to: {save_path}")


def generate_summary_report(metrics, class_names, save_path):
    """Generate a text summary report"""
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("BRAIN TUMOR DETECTION - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-" * 70 + "\n")
        
        metric_names = {
            'metrics/precision(B)': 'Precision (Box)',
            'metrics/recall(B)': 'Recall (Box)',
            'metrics/mAP50(B)': 'mAP@0.5',
            'metrics/mAP50-95(B)': 'mAP@0.5:0.95',
        }
        
        for key, name in metric_names.items():
            if key in metrics:
                value = metrics[key]
                f.write(f"{name:<25}: {value:.4f}\n")
        
        # Per-class metrics
        if class_names:
            f.write("\n\nPER-CLASS METRICS:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}\n")
            f.write("-" * 70 + "\n")
            
            for cls_id, cls_name in class_names.items():
                precision_key = f'metrics/precision(B)_{cls_id}'
                recall_key = f'metrics/recall(B)_{cls_id}'
                map50_key = f'metrics/mAP50(B)_{cls_id}'
                map50_95_key = f'metrics/mAP50-95(B)_{cls_id}'
                
                precision = metrics.get(precision_key, 0.0)
                recall = metrics.get(recall_key, 0.0)
                map50 = metrics.get(map50_key, 0.0)
                map50_95 = metrics.get(map50_95_key, 0.0)
                
                f.write(f"{cls_name:<20} {precision:<12.4f} {recall:<12.4f} {map50:<12.4f} {map50_95:<12.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ Summary report saved to: {save_path}")


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    # Check if data config exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Data configuration file not found: {args.data}")
        return
    
    # Load class names
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config.get('names', {})
    
    # Print header
    print_evaluation_header(args)
    
    # Load model
    print("📦 Loading model...")
    model = YOLO(args.model)
    print(f"✓ Model loaded: {args.model}")
    
    if class_names:
        print(f"✓ Classes: {', '.join(class_names.values())}")
    
    # Run evaluation
    print(f"\n🔍 Evaluating on {args.split} set...")
    
    try:
        # Validate model
        metrics = model.val(
            data=args.data,
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save_json=args.save_json,
            save_hybrid=args.save_hybrid,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            plots=args.plots,
            verbose=True,
        )
        
        # Get metrics dictionary
        if hasattr(metrics, 'results_dict'):
            metrics_dict = metrics.results_dict
        else:
            metrics_dict = {}
        
        # Print metrics
        print_metrics_summary(metrics_dict, class_names)
        print_confusion_matrix_summary(metrics_dict)
        
        # Save results
        save_dir = Path(args.project) / args.name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics to JSON
        if args.save_json or metrics_dict:
            json_path = save_dir / 'metrics.json'
            save_metrics_to_json(metrics_dict, json_path)
        
        # Generate summary report
        report_path = save_dir / 'evaluation_report.txt'
        generate_summary_report(metrics_dict, class_names, report_path)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to: {save_dir}")
        
        if args.plots:
            print(f"📊 Plots saved in: {save_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

# Made with Bob
