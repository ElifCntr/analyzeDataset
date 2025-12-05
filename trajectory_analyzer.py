"""Trajectory Analyzer - Reusable module for motion analysis"""
import numpy as np


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
        x1_min, y1_min = cx1 - w1 / 2, cy1 - h1 / 2
        x1_max, y1_max = cx1 + w1 / 2, cy1 + h1 / 2
        x2_min, y2_min = cx2 - w2 / 2, cy2 - h2 / 2
        x2_max, y2_max = cx2 + w2 / 2, cy2 + h2 / 2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def extract_trajectories(self, frame_annotations):
        """
        Extract trajectories by tracking objects across frames.

        Args:
            frame_annotations: List of frame annotations
                Each frame is a list of dicts with keys: class_id, cx, cy, w, h

        Returns:
            List of trajectory dicts with keys: frames, positions, boxes, length
        """
        trajectories, active = [], []

        for frame_idx, frame_anns in enumerate(frame_annotations):
            # Get objects of target class
            objects = [a for a in frame_anns if a['class_id'] == self.class_id]
            matched_traj, matched_obj = set(), set()

            # Match with active trajectories
            for ti, traj in enumerate(active):
                if ti in matched_traj:
                    continue
                if frame_idx - traj['frames'][-1] > self.max_frames_skip:
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
                    matched_traj.add(ti)
                    matched_obj.add(best_obj)

            # Create new trajectories for unmatched objects
            for oi, obj in enumerate(objects):
                if oi not in matched_obj:
                    active.append({
                        'frames': [frame_idx],
                        'positions': [(obj['cx'], obj['cy'])],
                        'boxes': [(obj['cx'], obj['cy'], obj['w'], obj['h'])]
                    })

            # Finalize old trajectories
            new_active = []
            for traj in active:
                if frame_idx - traj['frames'][-1] <= self.max_frames_skip:
                    new_active.append(traj)
                else:
                    traj['length'] = len(traj['frames'])
                    if traj['length'] >= 3:  # Keep trajectories with 3+ frames
                        trajectories.append(traj)
            active = new_active

        # Finalize remaining trajectories
        for traj in active:
            traj['length'] = len(traj['frames'])
            if traj['length'] >= 3:
                trajectories.append(traj)

        return trajectories

    def compute_trajectory_features(self, trajectory):
        """
        Compute motion features for a single trajectory.

        Returns:
            Dict with: length, avg_speed, speed_variance, is_hovering,
                       direction_changes, straightness
        """
        positions = np.array(trajectory['positions'])
        frames = np.array(trajectory['frames'])

        # Compute displacements and speeds
        displacements = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        avg_speed = np.mean(displacements) if len(displacements) > 0 else 0
        speed_variance = np.var(displacements) if len(displacements) > 0 else 0

        # Hovering detection (low speed variance + low average speed)
        is_hovering = speed_variance < 0.001 and avg_speed < 0.05

        # Direction changes
        direction_changes = 0
        if len(positions) >= 3:
            directions = np.diff(positions, axis=0)
            angles = np.arctan2(directions[:, 1], directions[:, 0])
            angle_diffs = np.abs(np.diff(angles))
            angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
            direction_changes = np.sum(angle_diffs > np.pi / 6)  # >30 degrees

        # Straightness (0=curved, 1=straight line)
        direct_dist = np.sqrt(np.sum((positions[-1] - positions[0]) ** 2))
        path_length = np.sum(displacements)
        straightness = direct_dist / path_length if path_length > 0 else 0

        return {
            'length': len(trajectory['frames']),
            'avg_speed': float(avg_speed),
            'speed_variance': float(speed_variance),
            'is_hovering': bool(is_hovering),
            'direction_changes': int(direction_changes),
            'straightness': float(straightness),
        }

    def analyze_trajectories(self, trajectories):
        """
        Compute aggregate statistics over all trajectories.

        Returns:
            Dict with: num_trajectories, avg_length, length_distribution,
                       avg_speed, hovering_percentage, avg_direction_changes,
                       avg_straightness
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
            'hovering_percentage': float((hovering_count / len(features)) * 100),
            'avg_direction_changes': float(np.mean([f['direction_changes'] for f in features])),
            'avg_straightness': float(np.mean([f['straightness'] for f in features])),
        }