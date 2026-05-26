"""
Brain Tumor Dataset Validator and Statistics Generator
Validates YOLO format dataset and provides comprehensive statistics
"""

import os
import sys
import glob
from pathlib import Path
from collections import defaultdict
import yaml

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class DatasetValidator:
    def __init__(self, data_yaml_path='data.yaml'):
        """Initialize validator with dataset configuration"""
        with open(data_yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_path = Path(self.config['path'])
        self.train_path = self.dataset_path / self.config['train']
        self.val_path = self.dataset_path / self.config['val']
        self.class_names = self.config['names']
        self.nc = self.config['nc']
        
    def validate_structure(self):
        """Validate dataset directory structure"""
        print("=" * 70)
        print("DATASET STRUCTURE VALIDATION")
        print("=" * 70)
        
        issues = []
        
        # Check if main directories exist
        if not self.train_path.exists():
            issues.append(f"Training directory not found: {self.train_path}")
        else:
            print(f"✓ Training directory found: {self.train_path}")
            
        if not self.val_path.exists():
            issues.append(f"Validation directory not found: {self.val_path}")
        else:
            print(f"✓ Validation directory found: {self.val_path}")
        
        return issues
    
    def count_files(self, split='train'):
        """Count images and labels for each class"""
        split_path = self.train_path if split == 'train' else self.val_path
        
        stats = defaultdict(lambda: {'images': 0, 'labels': 0, 'annotations': 0})
        
        # Iterate through each class directory
        for class_id, class_name in self.class_names.items():
            class_dir = split_path / class_name
            
            if not class_dir.exists():
                continue
            
            # Count images
            images_dir = class_dir / 'images'
            if images_dir.exists():
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                stats[class_name]['images'] = len(image_files)
            
            # Count labels and annotations
            labels_dir = class_dir / 'labels'
            if labels_dir.exists():
                label_files = list(labels_dir.glob('*.txt'))
                stats[class_name]['labels'] = len(label_files)
                
                # Count total annotations
                for label_file in label_files:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        stats[class_name]['annotations'] += len(lines)
        
        return stats
    
    def validate_labels(self, split='train', sample_size=10):
        """Validate label format and check for issues"""
        split_path = self.train_path if split == 'train' else self.val_path
        
        issues = []
        checked = 0
        
        for class_id, class_name in self.class_names.items():
            labels_dir = split_path / class_name / 'labels'
            
            if not labels_dir.exists():
                continue
            
            label_files = list(labels_dir.glob('*.txt'))[:sample_size]
            
            for label_file in label_files:
                checked += 1
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        
                        if len(parts) != 5:
                            issues.append(
                                f"{label_file.name} line {line_num}: "
                                f"Expected 5 values, got {len(parts)}"
                            )
                            continue
                        
                        try:
                            cls_id = int(parts[0])
                            x, y, w, h = map(float, parts[1:])
                            
                            # Validate class ID
                            if cls_id not in range(self.nc):
                                issues.append(
                                    f"{label_file.name} line {line_num}: "
                                    f"Invalid class ID {cls_id} (expected 0-{self.nc-1})"
                                )
                            
                            # Validate coordinates (should be normalized 0-1)
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                issues.append(
                                    f"{label_file.name} line {line_num}: "
                                    f"Coordinates out of range [0,1]: x={x}, y={y}, w={w}, h={h}"
                                )
                        
                        except ValueError as e:
                            issues.append(
                                f"{label_file.name} line {line_num}: "
                                f"Invalid format - {str(e)}"
                            )
        
        return issues, checked
    
    def print_statistics(self):
        """Print comprehensive dataset statistics"""
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        
        # Training set statistics
        print("\n📊 TRAINING SET:")
        print("-" * 70)
        train_stats = self.count_files('train')
        
        total_train_images = 0
        total_train_labels = 0
        total_train_annotations = 0
        
        print(f"{'Class':<20} {'Images':<10} {'Labels':<10} {'Annotations':<15}")
        print("-" * 70)
        
        for class_name in self.class_names.values():
            stats = train_stats[class_name]
            print(f"{class_name:<20} {stats['images']:<10} {stats['labels']:<10} {stats['annotations']:<15}")
            total_train_images += stats['images']
            total_train_labels += stats['labels']
            total_train_annotations += stats['annotations']
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_train_images:<10} {total_train_labels:<10} {total_train_annotations:<15}")
        
        # Validation set statistics
        print("\n📊 VALIDATION SET:")
        print("-" * 70)
        val_stats = self.count_files('val')
        
        total_val_images = 0
        total_val_labels = 0
        total_val_annotations = 0
        
        print(f"{'Class':<20} {'Images':<10} {'Labels':<10} {'Annotations':<15}")
        print("-" * 70)
        
        for class_name in self.class_names.values():
            stats = val_stats[class_name]
            print(f"{class_name:<20} {stats['images']:<10} {stats['labels']:<10} {stats['annotations']:<15}")
            total_val_images += stats['images']
            total_val_labels += stats['labels']
            total_val_annotations += stats['annotations']
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_val_images:<10} {total_val_labels:<10} {total_val_annotations:<15}")
        
        # Overall statistics
        print("\n📈 OVERALL STATISTICS:")
        print("-" * 70)
        print(f"Total Images:       {total_train_images + total_val_images}")
        print(f"Total Labels:       {total_train_labels + total_val_labels}")
        print(f"Total Annotations:  {total_train_annotations + total_val_annotations}")
        print(f"Number of Classes:  {self.nc}")
        print(f"Train/Val Split:    {total_train_images}/{total_val_images} "
              f"({total_train_images/(total_train_images+total_val_images)*100:.1f}%/"
              f"{total_val_images/(total_train_images+total_val_images)*100:.1f}%)")
    
    def run_validation(self):
        """Run complete validation"""
        print("\n[*] Brain Tumor Dataset Validation\n")
        
        # Validate structure
        structure_issues = self.validate_structure()
        
        if structure_issues:
            print("\n❌ STRUCTURE ISSUES FOUND:")
            for issue in structure_issues:
                print(f"  - {issue}")
            return
        
        # Print statistics
        self.print_statistics()
        
        # Validate labels
        print("\n" + "=" * 70)
        print("LABEL FORMAT VALIDATION")
        print("=" * 70)
        
        print("\n🔍 Checking training labels (sample)...")
        train_issues, train_checked = self.validate_labels('train', sample_size=20)
        print(f"Checked {train_checked} label files")
        
        print("\n🔍 Checking validation labels (sample)...")
        val_issues, val_checked = self.validate_labels('val', sample_size=20)
        print(f"Checked {val_checked} label files")
        
        all_issues = train_issues + val_issues
        
        if all_issues:
            print(f"\n⚠️  FOUND {len(all_issues)} ISSUES:")
            for issue in all_issues[:10]:  # Show first 10 issues
                print(f"  - {issue}")
            if len(all_issues) > 10:
                print(f"  ... and {len(all_issues) - 10} more issues")
        else:
            print("\n✅ All checked labels are valid!")
        
        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        if not structure_issues and not all_issues:
            print("✅ Dataset is ready for training!")
        else:
            print("⚠️  Please fix the issues above before training")
        
        print("=" * 70)


def main():
    """Main function"""
    try:
        validator = DatasetValidator('data.yaml')
        validator.run_validation()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure data.yaml exists in the current directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()

# Made with Bob
