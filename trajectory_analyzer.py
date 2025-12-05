"""Trajectory Analyzer - Extract and analyze object trajectories"""
import numpy as np
from typing import List, Dict


class TrajectoryAnalyzer:
    """Extract and analyze object trajectories from frame annotations."""
    
    def __init__(self, class_id=0, iou_threshold=0.5, max_frames_skip=5):
        """
        Args:
            class_id: Class to track (0=drone, 1=bird)
            iou_threshold: IoU threshold for matching objects across frames
            max_frames_skip: Max frames gap before ending trajectory
        """
        self.class_id = class_id
        self.iou_threshold = iou_threshold
        self.max_frames_skip = max_frames_skip
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes (cx, cy, w, h)."""
        cx1, cy1, w1, h1 = box1
        cx2, cy2, w2, h2 = box2
        x1_min, y1_min = cx1 - w1/2, cy1 - h1/2
        x1_max, y1_max = cx1 + w1/2, cy1 + h1/2
        x2_min, y2_min = cx2 - w2/2, cy2 - h2/2
        x2_max, y2_max = cx2 + w2/2, cy2 + h2/2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        union_area = w1*h1 + w2*h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
    
    def extract_trajectories(self, frame_annotations, img_width=1920, img_height=1080):
        """
        Extract trajectories by tracking objects across frames.
        
        Args:
            frame_annotations: List of frame annotations
                Each frame is a list of dicts with keys: class_id, cx, cy, w, h
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            List of trajectory dicts with keys: frames, positions, boxes, length, widths, heights, areas, aspect_ratios
        """
        trajectories, active = [], []
        
        for frame_idx, frame_anns in enumerate(frame_annotations):
            objects = [a for a in frame_anns if a['class_id'] == self.class_id]
            matched_traj, matched_obj = set(), set()
            
            # Match with active trajectories
            for ti, traj in enumerate(active):
                if ti in matched_traj or frame_idx - traj['frames'][-1] > self.max_frames_skip:
                    continue
                
                last_box = traj['boxes'][-1]
                best_iou, best_obj = 0, None
                
                for oi, obj in enumerate(objects):
                    if oi in matched_obj:
                        continue
                    iou = self.compute_iou(last_box, (obj['cx'], obj['cy'], obj['w'], obj['h']))
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou, best_obj = iou, oi
                
                if best_obj is not None:
                    obj = objects[best_obj]
                    traj['frames'].append(frame_idx)
                    traj['positions'].append((obj['cx'], obj['cy']))
                    traj['boxes'].append((obj['cx'], obj['cy'], obj['w'], obj['h']))
                    traj['widths'].append(obj['w'] * img_width)
                    traj['heights'].append(obj['h'] * img_height)
                    traj['areas'].append(obj['w'] * img_width * obj['h'] * img_height)
                    traj['aspect_ratios'].append(obj['w'] / obj['h'] if obj['h'] > 0 else 1.0)
                    matched_traj.add(ti)
                    matched_obj.add(best_obj)
            
            # Create new trajectories
            for oi, obj in enumerate(objects):
                if oi not in matched_obj:
                    active.append({
                        'frames': [frame_idx],
                        'positions': [(obj['cx'], obj['cy'])],
                        'boxes': [(obj['cx'], obj['cy'], obj['w'], obj['h'])],
                        'widths': [obj['w'] * img_width],
                        'heights': [obj['h'] * img_height],
                        'areas': [obj['w'] * img_width * obj['h'] * img_height],
                        'aspect_ratios': [obj['w'] / obj['h'] if obj['h'] > 0 else 1.0],
                    })
            
            # Finalize old trajectories
            new_active = []
            for traj in active:
                if frame_idx - traj['frames'][-1] <= self.max_frames_skip:
                    new_active.append(traj)
                else:
                    traj['length'] = len(traj['frames'])
                    if traj['length'] >= 3:
                        trajectories.append(traj)
            active = new_active
        
        # Finalize remaining
        for traj in active:
            traj['length'] = len(traj['frames'])
            if traj['length'] >= 3:
                trajectories.append(traj)
        
        return trajectories
    
    def compute_trajectory_features(self, trajectory):
        """
        Compute motion and appearance features for a single trajectory.
        
        Returns:
            Dict with motion and appearance features
        """
        positions = np.array(trajectory['positions'])
        frames = np.array(trajectory['frames'])
        widths = np.array(trajectory['widths'])
        heights = np.array(trajectory['heights'])
        areas = np.array(trajectory['areas'])
        aspect_ratios = np.array(trajectory['aspect_ratios'])
        
        # === MOTION FEATURES ===
        displacements = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        avg_speed = np.mean(displacements) if len(displacements) > 0 else 0
        max_speed = np.max(displacements) if len(displacements) > 0 else 0
        speed_variance = np.var(displacements) if len(displacements) > 0 else 0
        speed_std = np.std(displacements) if len(displacements) > 0 else 0
        
        # Hovering detection
        is_hovering = avg_speed < 0.02
        
        # Direction changes
        direction_changes = 0
        if len(positions) >= 3:
            directions = np.diff(positions, axis=0)
            angles = np.arctan2(directions[:, 1], directions[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
            direction_changes = np.sum(angle_diffs > np.pi/6)
        
        # Straightness
        direct_dist = np.sqrt(np.sum((positions[-1] - positions[0])**2))
        path_length = np.sum(displacements)
        straightness = direct_dist / path_length if path_length > 0 else 0
        
        # Acceleration
        if len(displacements) >= 2:
            accelerations = np.diff(displacements)
            avg_acceleration = np.mean(np.abs(accelerations))
            max_acceleration = np.max(np.abs(accelerations))
        else:
            avg_acceleration = 0
            max_acceleration = 0
        
        # === APPEARANCE FEATURES ===
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        avg_area = np.mean(areas)
        
        width_variance = np.var(widths)
        height_variance = np.var(heights)
        area_variance = np.var(areas)
        
        # Relative size change
        if len(areas) > 1:
            area_changes = np.diff(areas)
            relative_area_changes = area_changes / areas[:-1]
            avg_relative_area_change = np.mean(np.abs(relative_area_changes))
            max_relative_area_change = np.max(np.abs(relative_area_changes))
        else:
            avg_relative_area_change = 0
            max_relative_area_change = 0
        
        # Aspect ratio stability
        avg_aspect_ratio = np.mean(aspect_ratios)
        aspect_ratio_variance = np.var(aspect_ratios)
        aspect_ratio_stable = aspect_ratio_variance < 0.01
        
        # Scale trend
        if len(areas) >= 2:
            scale_trend = (areas[-1] - areas[0]) / areas[0] if areas[0] > 0 else 0
            is_approaching = scale_trend > 0.1
            is_receding = scale_trend < -0.1
        else:
            scale_trend = 0
            is_approaching = False
            is_receding = False
        
        return {
            'length': len(trajectory['frames']),
            'duration': frames[-1] - frames[0] + 1,
            'avg_speed': float(avg_speed),
            'max_speed': float(max_speed),
            'speed_variance': float(speed_variance),
            'speed_std': float(speed_std),
            'is_hovering': bool(is_hovering),
            'direction_changes': int(direction_changes),
            'straightness': float(straightness),
            'avg_acceleration': float(avg_acceleration),
            'max_acceleration': float(max_acceleration),
            'avg_width': float(avg_width),
            'avg_height': float(avg_height),
            'avg_area': float(avg_area),
            'width_variance': float(width_variance),
            'height_variance': float(height_variance),
            'area_variance': float(area_variance),
            'avg_relative_area_change': float(avg_relative_area_change),
            'max_relative_area_change': float(max_relative_area_change),
            'avg_aspect_ratio': float(avg_aspect_ratio),
            'aspect_ratio_variance': float(aspect_ratio_variance),
            'aspect_ratio_stable': bool(aspect_ratio_stable),
            'scale_trend': float(scale_trend),
            'is_approaching': bool(is_approaching),
            'is_receding': bool(is_receding),
        }
    
    def analyze_trajectories(self, trajectories):
        """
        Compute aggregate statistics over all trajectories.
        
        Returns:
            Dict with comprehensive trajectory statistics
        """
        if not trajectories:
            return {
                'num_trajectories': 0,
                'avg_length': 0,
                'length_distribution': {'short': 0, 'medium': 0, 'long': 0},
            }
        
        features = [self.compute_trajectory_features(t) for t in trajectories]
        lengths = [f['length'] for f in features]
        
        hovering_count = sum(1 for f in features if f['is_hovering'])
        approaching_count = sum(1 for f in features if f['is_approaching'])
        receding_count = sum(1 for f in features if f['is_receding'])
        stable_shape_count = sum(1 for f in features if f['aspect_ratio_stable'])
        
        return {
            'num_trajectories': len(trajectories),
            'avg_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'length_distribution': {
                'short': sum(1 for l in lengths if l < 30),
                'medium': sum(1 for l in lengths if 30 <= l <= 100),
                'long': sum(1 for l in lengths if l > 100),
            },
            'avg_speed': float(np.mean([f['avg_speed'] for f in features])),
            'max_speed_overall': float(np.max([f['max_speed'] for f in features])),
            'avg_speed_variance': float(np.mean([f['speed_variance'] for f in features])),
            'hovering_count': int(hovering_count),
            'hovering_percentage': float((hovering_count / len(features)) * 100),
            'avg_direction_changes': float(np.mean([f['direction_changes'] for f in features])),
            'avg_straightness': float(np.mean([f['straightness'] for f in features])),
            'avg_acceleration': float(np.mean([f['avg_acceleration'] for f in features])),
            'avg_object_width': float(np.mean([f['avg_width'] for f in features])),
            'avg_object_height': float(np.mean([f['avg_height'] for f in features])),
            'avg_object_area': float(np.mean([f['avg_area'] for f in features])),
            'avg_area_variance': float(np.mean([f['area_variance'] for f in features])),
            'avg_relative_area_change': float(np.mean([f['avg_relative_area_change'] for f in features])),
            'avg_aspect_ratio': float(np.mean([f['avg_aspect_ratio'] for f in features])),
            'stable_shape_count': int(stable_shape_count),
            'stable_shape_percentage': float((stable_shape_count / len(features)) * 100),
            'approaching_count': int(approaching_count),
            'approaching_percentage': float((approaching_count / len(features)) * 100),
            'receding_count': int(receding_count),
            'receding_percentage': float((receding_count / len(features)) * 100),
        }
