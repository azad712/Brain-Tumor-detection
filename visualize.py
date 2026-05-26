"""
Brain Tumor Detection - Visualization Script
Visualizes predictions with bounding boxes and confidence scores
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml
from ultralytics import YOLO


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize YOLOv8 Brain Tumor Detections')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image or directory of images')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default='visualizations',
                        help='Output directory for visualizations')
    parser.add_argument('--show_conf', action='store_true', default=True,
                        help='Show confidence scores on boxes')
    parser.add_argument('--show_labels', action='store_true', default=True,
                        help='Show class labels on boxes')
    parser.add_argument('--line_width', type=int, default=2,
                        help='Bounding box line width')
    parser.add_argument('--font_size', type=int, default=12,
                        help='Font size for labels')
    parser.add_argument('--save_format', type=str, default='jpg',
                        choices=['jpg', 'png', 'pdf'],
                        help='Output image format')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved images')
    parser.add_argument('--max_images', type=int, default=50,
                        help='Maximum number of images to process')
    parser.add_argument('--grid', action='store_true',
                        help='Create grid visualization of multiple images')
    parser.add_argument('--grid_cols', type=int, default=4,
                        help='Number of columns in grid visualization')
    
    return parser.parse_args()


def load_class_names(model_path):
    """Load class names from model or data.yaml"""
    try:
        model = YOLO(model_path)
        if hasattr(model, 'names'):
            return model.names
    except:
        pass
    
    if os.path.exists('data.yaml'):
        with open('data.yaml', 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', {})
    
    return {}


def get_color_map(num_classes):
    """Generate distinct colors for each class"""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = plt.cm.hsv(hue)[:3]
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def draw_boxes_cv2(image, boxes, class_names, colors, show_conf=True, show_labels=True, line_width=2):
    """Draw bounding boxes on image using OpenCV"""
    img = image.copy()
    
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        
        # Get class name and color
        class_name = class_names.get(cls_id, f"Class_{cls_id}")
        color = colors[cls_id % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
        
        # Prepare label
        if show_labels and show_conf:
            label = f"{class_name} {conf:.2f}"
        elif show_labels:
            label = class_name
        elif show_conf:
            label = f"{conf:.2f}"
        else:
            label = ""
        
        # Draw label background and text
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text
            cv2.putText(img, label, (x1, y1 - baseline - 2), 
                       font, font_scale, (255, 255, 255), thickness)
    
    return img


def visualize_single_image(image_path, model, class_names, colors, args):
    """Visualize detections on a single image"""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️  Could not read image: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=args.conf,
        iou=args.iou,
        verbose=False
    )
    
    # Draw boxes
    if results and len(results) > 0 and results[0].boxes is not None:
        img_with_boxes = draw_boxes_cv2(
            img_rgb, 
            results[0].boxes, 
            class_names, 
            colors,
            show_conf=args.show_conf,
            show_labels=args.show_labels,
            line_width=args.line_width
        )
        return img_with_boxes, results[0]
    
    return img_rgb, None


def create_grid_visualization(images_data, args):
    """Create grid visualization of multiple images"""
    num_images = len(images_data)
    cols = args.grid_cols
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx, (img, img_path, result) in enumerate(images_data):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        ax.imshow(img)
        ax.axis('off')
        
        # Add title with filename and detection count
        filename = os.path.basename(img_path)
        if result and result.boxes is not None:
            num_detections = len(result.boxes)
            title = f"{filename}\n({num_detections} detections)"
        else:
            title = f"{filename}\n(No detections)"
        
        ax.set_title(title, fontsize=args.font_size - 2)
    
    # Hide empty subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    """Main visualization function"""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Error: Model file not found: {args.model}")
        return
    
    # Check if source exists
    if not os.path.exists(args.source):
        print(f"❌ Error: Source not found: {args.source}")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("BRAIN TUMOR DETECTION - VISUALIZATION")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Source:      {args.source}")
    print(f"Output:      {output_dir}")
    print(f"Confidence:  {args.conf}")
    print("=" * 70 + "\n")
    
    # Load model
    print("📦 Loading model...")
    model = YOLO(args.model)
    
    # Get class names and colors
    class_names = load_class_names(args.model)
    num_classes = len(class_names) if class_names else 4
    colors = get_color_map(num_classes)
    
    if class_names:
        print(f"✓ Classes: {', '.join(class_names.values())}")
    
    # Get image paths
    source_path = Path(args.source)
    if source_path.is_file():
        image_paths = [source_path]
    else:
        image_paths = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
        image_paths = image_paths[:args.max_images]
    
    if not image_paths:
        print("❌ No images found in source")
        return
    
    print(f"\n🔍 Processing {len(image_paths)} images...")
    
    # Process images
    images_data = []
    processed = 0
    
    for img_path in image_paths:
        img_with_boxes, result = visualize_single_image(
            img_path, model, class_names, colors, args
        )
        
        if img_with_boxes is not None:
            images_data.append((img_with_boxes, str(img_path), result))
            
            # Save individual image
            if not args.grid:
                output_path = output_dir / f"{img_path.stem}_detected.{args.save_format}"
                img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), img_bgr)
            
            processed += 1
            
            # Print progress
            if result and result.boxes is not None:
                num_detections = len(result.boxes)
                print(f"  ✓ {img_path.name}: {num_detections} detections")
            else:
                print(f"  ✓ {img_path.name}: No detections")
    
    # Create grid visualization if requested
    if args.grid and images_data:
        print("\n📊 Creating grid visualization...")
        fig = create_grid_visualization(images_data, args)
        grid_path = output_dir / f"grid_visualization.{args.save_format}"
        fig.savefig(grid_path, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Grid saved to: {grid_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION SUMMARY")
    print("=" * 70)
    print(f"Processed images:  {processed}/{len(image_paths)}")
    print(f"Output directory:  {output_dir}")
    print(f"Format:            {args.save_format}")
    print("=" * 70)
    
    print("\n✅ Visualization completed successfully!")


if __name__ == "__main__":
    main()

# Made with Bob
