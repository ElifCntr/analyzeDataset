"""
SussexDrone specific analyzer - extends base YOLODatasetAnalyzer
"""
from dataset_analyzer import YOLODatasetAnalyzer
from typing import Dict, Optional


class SussexDroneAnalyzer(YOLODatasetAnalyzer):
    """
    SussexDrone-specific analyzer with custom categorization.
    """
    
    def __init__(self, dataset_root: str):
        # SussexDrone has 3 classes
        class_names = {
            0: 'drone',
            1: 'bird',
            2: 'other'
        }
        
        super().__init__(dataset_root, class_names)
    
    def guess_capture_method(self, video_name: str, resolution: str) -> tuple:
        """
        Guess capture method from video name and resolution.
        Returns: (method, confidence)
        """
        name = video_name.lower()
        
        # High confidence patterns
        if 'drone_to_drone' in name or 'drone2drone' in name:
            return ('drone_to_drone', 'high')
        
        if 'dscf' in name:
            return ('static_ground', 'high')
        
        # Medium confidence
        if 'dynamic' in name:
            return ('dynamic_aerial', 'medium')
        
        if 'gx' in name or 'gopro' in name:
            return ('dynamic_aerial', 'medium')
        
        # Low confidence from resolution
        if resolution == '1920x1080':
            return ('static_ground', 'low')
        elif resolution == '3840x2160':
            return ('dynamic_aerial', 'low')
        
        return ('unknown', 'none')
    
    def categorize_content(self, stats: Dict) -> str:
        """Categorize video content based on objects present."""
        has_drone = stats.get('class_0_count', 0) > 0
        has_bird = stats.get('class_1_count', 0) > 0
        has_other = stats.get('class_2_count', 0) > 0
        
        if has_drone and has_bird:
            return 'mixed'
        elif has_drone:
            return 'drone_only'
        elif has_bird:
            return 'bird_only'
        elif has_other:
            return 'other_only'
        else:
            return 'background'
    
    def _compute_video_stats(self, video_name, resolution, total_frames, 
                            frame_data, class_counts, all_annotations):
        """Extended stats with SussexDrone-specific categorization."""
        stats = super()._compute_video_stats(
            video_name, resolution, total_frames, 
            frame_data, class_counts, all_annotations
        )
        
        # Add capture method
        capture_method, confidence = self.guess_capture_method(video_name, resolution)
        stats['capture_method'] = capture_method
        stats['capture_confidence'] = confidence
        
        # Add content category
        stats['content_category'] = self.categorize_content(stats)
        
        # Add mixed frames count
        stats['mixed_frames'] = sum(
            1 for f in frame_data 
            if f.get(0, 0) > 0 and f.get(1, 0) > 0
        )
        
        # Drone presence pattern
        drone_frames = [f.get(0, 0) for f in frame_data]
        max_drones = max(drone_frames) if drone_frames else 0
        
        if max_drones > 1:
            stats['drone_pattern'] = 'multiple'
        elif max_drones == 1:
            stats['drone_pattern'] = 'single'
        else:
            stats['drone_pattern'] = 'none'
        
        return stats
