"""
Base Dataset Analyzer - Reusable for any YOLO format dataset
"""
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple


class YOLODatasetAnalyzer:
    """
    Modular analyzer for YOLO format datasets.
    Easily configurable for different dataset structures.
    """
    
    def __init__(self, 
                 dataset_root: str,
                 class_names: Optional[Dict[int, str]] = None,
                 size_thresholds: Optional[Dict[str, int]] = None):
        """
        Initialize analyzer.
        
        Args:
            dataset_root: Path to dataset root
            class_names: Dict mapping class_id to name (e.g., {0: 'drone', 1: 'bird'})
            size_thresholds: Dict with 'small' and 'medium' pixel thresholds
        """
        self.dataset_root = Path(dataset_root)
        self.class_names = class_names or {}
        
        # MS-COCO standard thresholds
        self.size_thresholds = size_thresholds or {
            'small': 32 * 32,
            'medium': 96 * 96
        }
        
        self.videos = []
        
    def find_videos(self, 
                    image_folder_name: str = 'obj_train_data',
                    image_extensions: List[str] = ['.PNG', '.png', '.jpg', '.jpeg']) -> List[Path]:
        """
        Find all video folders in dataset.
        Override this method for different structures.
        
        Args:
            image_folder_name: Name of folder containing images
            image_extensions: Valid image file extensions
            
        Returns:
            List of video folder paths
        """
        video_folders = []
        
        for project in self.dataset_root.iterdir():
            if not project.is_dir() or project.name.startswith('.'):
                continue
                
            for video_folder in project.iterdir():
                if not video_folder.is_dir():
                    continue
                    
                img_folder = video_folder / image_folder_name
                if img_folder.exists():
                    video_folders.append(video_folder)
        
        return video_folders
    
    def parse_yolo_annotation(self, txt_path: Path, img_width: int, img_height: int) -> List[Dict]:
        """
        Parse YOLO format annotation file.
        
        Returns:
            List of annotation dicts with keys: class_id, cx, cy, w, h, pixel_area, size_category
        """
        if not txt_path.exists():
            return []
        
        annotations = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                
                # Convert to pixels
                pixel_width = w * img_width
                pixel_height = h * img_height
                pixel_area = pixel_width * pixel_height
                
                # Categorize size
                if pixel_area <= self.size_thresholds['small']:
                    size_category = 'small'
                elif pixel_area <= self.size_thresholds['medium']:
                    size_category = 'medium'
                else:
                    size_category = 'large'
                
                annotations.append({
                    'class_id': class_id,
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'h': h,
                    'pixel_width': pixel_width,
                    'pixel_height': pixel_height,
                    'pixel_area': pixel_area,
                    'size_category': size_category,
                })
        
        return annotations
    
    def get_image_size(self, img_path: Path) -> Tuple[int, int]:
        """Get image dimensions."""
        try:
            with Image.open(img_path) as img:
                return img.size
        except:
            return (1920, 1080)  # Default fallback
    
    def analyze_video(self, 
                     video_folder: Path,
                     image_folder_name: str = 'obj_train_data') -> Optional[Dict]:
        """
        Analyze a single video folder.
        
        Returns:
            Dict with video statistics
        """
        img_folder = video_folder / image_folder_name
        if not img_folder.exists():
            return None
        
        # Get images
        images = list(img_folder.glob('*.PNG')) + list(img_folder.glob('*.png'))
        images += list(img_folder.glob('*.jpg')) + list(img_folder.glob('*.jpeg'))
        
        if not images:
            return None
        
        # Get image size
        img_width, img_height = self.get_image_size(images[0])
        
        # Analyze annotations
        frame_data = []
        class_counts = Counter()
        all_annotations = []
        
        for img_path in images:
            txt_path = img_path.with_suffix('.txt')
            annotations = self.parse_yolo_annotation(txt_path, img_width, img_height)
            
            frame_info = {class_id: 0 for class_id in self.class_names.keys()}
            
            for ann in annotations:
                class_id = ann['class_id']
                class_counts[class_id] += 1
                frame_info[class_id] = frame_info.get(class_id, 0) + 1
                all_annotations.append(ann)
            
            frame_data.append(frame_info)
        
        # Compute statistics
        stats = self._compute_video_stats(
            video_folder.name,
            f"{img_width}x{img_height}",
            len(images),
            frame_data,
            class_counts,
            all_annotations
        )
        
        return stats
    
    def _compute_video_stats(self,
                            video_name: str,
                            resolution: str,
                            total_frames: int,
                            frame_data: List[Dict],
                            class_counts: Counter,
                            all_annotations: List[Dict]) -> Dict:
        """
        Compute statistics for a video.
        Override to add custom statistics.
        """
        stats = {
            'video_name': video_name,
            'resolution': resolution,
            'total_frames': total_frames,
            'class_counts': dict(class_counts),
        }
        
        # Per-class statistics
        for class_id in self.class_names.keys():
            class_anns = [a for a in all_annotations if a['class_id'] == class_id]
            
            if class_anns:
                sizes = [a['pixel_area'] for a in class_anns]
                stats[f'class_{class_id}_count'] = len(class_anns)
                stats[f'class_{class_id}_mean_size'] = np.mean(sizes)
                stats[f'class_{class_id}_small_count'] = sum(1 for a in class_anns if a['size_category'] == 'small')
                stats[f'class_{class_id}_medium_count'] = sum(1 for a in class_anns if a['size_category'] == 'medium')
                stats[f'class_{class_id}_large_count'] = sum(1 for a in class_anns if a['size_category'] == 'large')
            else:
                stats[f'class_{class_id}_count'] = 0
        
        # Frames with objects
        stats['annotated_frames'] = sum(1 for f in frame_data if any(f.values()))
        stats['empty_frames'] = total_frames - stats['annotated_frames']
        
        return stats
    
    def analyze_all(self, image_folder_name: str = 'obj_train_data') -> List[Dict]:
        """
        Analyze entire dataset.
        
        Returns:
            List of video statistics
        """
        video_folders = self.find_videos(image_folder_name)
        
        results = []
        for video_folder in video_folders:
            stats = self.analyze_video(video_folder, image_folder_name)
            if stats:
                results.append(stats)
        
        self.videos = results
        return results
    
    def get_overall_stats(self) -> Dict:
        """Get dataset-wide statistics."""
        if not self.videos:
            return {}
        
        overall = {
            'total_videos': len(self.videos),
            'total_frames': sum(v['total_frames'] for v in self.videos),
        }
        
        for class_id in self.class_names.keys():
            overall[f'total_class_{class_id}'] = sum(
                v.get(f'class_{class_id}_count', 0) for v in self.videos
            )
        
        return overall
