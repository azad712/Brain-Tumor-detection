"""
Brain Tumor Detection - YOLOv8 Inference Script
Performs inference on images using trained YOLOv8 model
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import yaml


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLOv8 Inference for Brain Tumor Detection')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image, directory, or video file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (e.g., 0 or cpu)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save inference results')
    parser.add_argument('--save_txt', action='store_true',
                        help='Save results to txt files')
    parser.add_argument('--save_conf', action='store_true',
                        help='Save confidences in txt files')
    parser.add_argument('--save_crop', action='store_true',
                        help='Save cropped prediction boxes')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--project', type=str, default='runs/predict',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--exist_ok', action='store_true',
                        help='Allow overwriting existing project/name')
    parser.add_argument('--line_width', type=int, default=2,
                        help='Bounding box line width')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize model features')
    parser.add_argument('--augment', action='store_true',
                        help='Use augmented inference')
    parser.add_argument('--agnostic_nms', action='store_true',
                        help='Class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='Filter by class (e.g., --classes 0 1 2)')
    parser.add_argument('--max_det', type=int, default=300,
                        help='Maximum number of detections per image')
    
    return parser.parse_args()


def load_class_names(model_path):
    """Load class names from model or data.yaml"""
    try:
        # Try to get class names from model
        model = YOLO(model_path)
        if hasattr(model, 'names'):
            return model.names
    except:
        pass
    
    # Fallback to data.yaml
    if os.path.exists('data.yaml'):
        with open('data.yaml', 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', {})
    
    return {}


def print_detection_summary(results, class_names):
    """Print summary of detections"""
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    
    total_detections = 0
    class_counts = {}
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes
            total_detections += len(boxes)
            
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = class_names.get(cls_id, f"Class_{cls_id}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"Total detections: {total_detections}")
    
    if class_counts:
        print("\nDetections by class:")
        print("-" * 70)
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name:<20}: {count}")
    else:
        print("\nNo detections found")
    
    print("=" * 70)


def print_detailed_results(results, class_names):
    """Print detailed detection results"""
    print("\n" + "=" * 70)
    print("DETAILED DETECTION RESULTS")
    print("=" * 70)
    
    for idx, result in enumerate(results):
        img_path = result.path
        print(f"\nImage {idx + 1}: {os.path.basename(img_path)}")
        print("-" * 70)
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            print(f"{'Class':<20} {'Confidence':<12} {'Bounding Box (x1, y1, x2, y2)'}")
            print("-" * 70)
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                class_name = class_names.get(cls_id, f"Class_{cls_id}")
                bbox_str = f"({xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f})"
                
                print(f"{class_name:<20} {conf:<12.4f} {bbox_str}")
        else:
            print("No detections")
    
    print("=" * 70)


def main():
    """Main inference function"""
    # Parse arguments
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    # Check if source exists
    if not os.path.exists(args.source):
        print(f"❌ Error: Source not found: {args.source}")
        return
    
    print("\n" + "=" * 70)
    print("BRAIN TUMOR DETECTION - INFERENCE")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Source:      {args.source}")
    print(f"Confidence:  {args.conf}")
    print(f"IoU:         {args.iou}")
    print(f"Image size:  {args.imgsz}")
    print(f"Device:      {args.device if args.device else 'auto'}")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading model...")
    model = YOLO(args.model)
    
    # Get class names
    class_names = load_class_names(args.model)
    if class_names:
        print(f"✓ Classes: {', '.join(class_names.values())}")
    
    # Run inference
    print(f"\n🔍 Running inference on: {args.source}")
    
    try:
        results = model.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            show=args.show,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
            line_width=args.line_width,
            visualize=args.visualize,
            augment=args.augment,
            agnostic_nms=args.agnostic_nms,
            classes=args.classes,
            max_det=args.max_det,
            verbose=True,
        )
        
        # Print results
        print_detection_summary(results, class_names)
        print_detailed_results(results, class_names)
        
        # Print save location
        if args.save:
            save_dir = Path(args.project) / args.name
            print(f"\n✅ Results saved to: {save_dir}")
            print(f"   - Annotated images in: {save_dir}")
            if args.save_txt:
                print(f"   - Label files in: {save_dir / 'labels'}")
            if args.save_crop:
                print(f"   - Cropped detections in: {save_dir / 'crops'}")
        
        print("\n✅ Inference completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Inference interrupted by user")
    except Exception as e:
        print(f"\n❌ Inference failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

# Made with Bob
