"""
Shape Analyzer
Analyzes object shape and appearance directly from bounding box regions
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class ShapeAnalyzer:
    """Analyze object shape and appearance from bounding box regions."""
    
    def __init__(self):
        pass
    
    def extract_region(self, frame, bbox):
        """
        Extract region from bounding box.
        
        Args:
            frame: Full frame image
            bbox: (cx, cy, w, h) in normalized coordinates
            
        Returns:
            Cropped region as numpy array
        """
        h, w = frame.shape[:2]
        cx, cy, box_w, box_h = bbox
        
        cx_px = int(cx * w)
        cy_px = int(cy * h)
        w_px = int(box_w * w)
        h_px = int(box_h * h)
        
        x1 = max(0, cx_px - w_px // 2)
        y1 = max(0, cy_px - h_px // 2)
        x2 = min(w, cx_px + w_px // 2)
        y2 = min(h, cy_px + h_px // 2)
        
        roi = frame[y1:y2, x1:x2]
        return roi if roi.size > 0 else None
    
    def analyze_shape(self, roi) -> Optional[Dict]:
        """
        Analyze shape and appearance features.
        
        Args:
            roi: Cropped region
            
        Returns:
            Dict with texture, color, edge, and shape features
        """
        if roi is None or roi.size == 0:
            return None
        
        h, w = roi.shape[:2]
        if h < 5 or w < 5:
            return None
        
        # Grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # === PIXEL STATISTICS ===
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        min_intensity = np.min(gray)
        max_intensity = np.max(gray)
        
        # === EDGE DETECTION ===
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        edge_count = np.sum(edges > 0)
        
        # === TEXTURE ===
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(gradient_magnitude)
        std_gradient = np.std(gradient_magnitude)
        
        # === FOREGROUND DETECTION ===
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground_pixel_count = np.sum(binary > 0)
        foreground_ratio = foreground_pixel_count / binary.size
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            contour_perimeter = cv2.arcLength(largest_contour, True)
            
            if contour_area > 10 and contour_perimeter > 0:
                # Circularity
                circularity = (4 * np.pi * contour_area) / (contour_perimeter ** 2)
                
                # Convex hull
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                solidity = contour_area / hull_area if hull_area > 0 else 0
                
                # Bounding rect
                x, y, cw, ch = cv2.boundingRect(largest_contour)
                rect_area = cw * ch
                extent = contour_area / rect_area if rect_area > 0 else 0
                aspect_ratio = float(cw) / ch if ch > 0 else 0
                
                # Fit ellipse
                if len(largest_contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(largest_contour)
                        (ecx, ecy), (ma, MA), angle = ellipse
                        eccentricity = np.sqrt(1 - (ma / MA) ** 2) if MA > 0 else 0
                        orientation = angle
                    except:
                        eccentricity = 0
                        orientation = 0
                else:
                    eccentricity = 0
                    orientation = 0
                
                compactness = (contour_perimeter ** 2) / contour_area if contour_area > 0 else 0
            else:
                circularity = solidity = extent = aspect_ratio = 0
                eccentricity = orientation = compactness = 0
        else:
            contour_area = contour_perimeter = 0
            circularity = solidity = extent = aspect_ratio = 0
            eccentricity = orientation = compactness = 0
        
        # === COLOR ===
        if len(roi.shape) == 3:
            mean_b, mean_g, mean_r = np.mean(roi[:, :, 0]), np.mean(roi[:, :, 1]), np.mean(roi[:, :, 2])
            std_b, std_g, std_r = np.std(roi[:, :, 0]), np.std(roi[:, :, 1]), np.std(roi[:, :, 2])
        else:
            mean_b = mean_g = mean_r = mean_intensity
            std_b = std_g = std_r = std_intensity
        
        # === HISTOGRAM ===
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = hist.flatten() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return {
            'mean_intensity': float(mean_intensity),
            'std_intensity': float(std_intensity),
            'intensity_range': float(max_intensity - min_intensity),
            'mean_gradient': float(mean_gradient),
            'std_gradient': float(std_gradient),
            'edge_density': float(edge_density),
            'edge_count': int(edge_count),
            'entropy': float(entropy),
            'foreground_pixel_count': int(foreground_pixel_count),
            'foreground_ratio': float(foreground_ratio),
            'contour_area': float(contour_area),
            'contour_perimeter': float(contour_perimeter),
            'circularity': float(circularity),
            'solidity': float(solidity),
            'extent': float(extent),
            'contour_aspect_ratio': float(aspect_ratio),
            'eccentricity': float(eccentricity),
            'orientation': float(orientation),
            'compactness': float(compactness),
            'mean_blue': float(mean_b),
            'mean_green': float(mean_g),
            'mean_red': float(mean_r),
            'std_blue': float(std_b),
            'std_green': float(std_g),
            'std_red': float(std_r),
            'region_width': int(w),
            'region_height': int(h),
            'region_area': int(w * h),
        }
    
    def analyze_trajectory_shapes(self, video_folder: Path, trajectory: Dict,
                                   img_folder_name='obj_train_data') -> List[Dict]:
        """
        Analyze shapes for each frame in a trajectory.
        
        Args:
            video_folder: Path to video folder
            trajectory: Trajectory dict with 'frames' and 'boxes'
            img_folder_name: Name of image folder
            
        Returns:
            List of shape features for each frame
        """
        img_folder = video_folder / img_folder_name
        if not img_folder.exists():
            return []
        
        all_images = sorted(list(img_folder.glob('*.PNG')) + list(img_folder.glob('*.png')))
        if not all_images:
            return []
        
        shape_features = []
        
        for i, frame_idx in enumerate(trajectory['frames']):
            if frame_idx >= len(all_images):
                continue
            
            img_path = all_images[frame_idx]
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            bbox = trajectory['boxes'][i]
            roi = self.extract_region(frame, bbox)
            
            if roi is not None:
                features = self.analyze_shape(roi)
                if features is not None:
                    shape_features.append(features)
        
        return shape_features
    
    def compute_shape_change_statistics(self, shape_features: List[Dict]) -> Dict:
        """
        Compute statistics about shape changes over trajectory.
        
        Returns:
            Dict with mean, variance, and change metrics
        """
        if not shape_features:
            return {}
        
        # Time series
        intensities = [f['mean_intensity'] for f in shape_features]
        gradients = [f['mean_gradient'] for f in shape_features]
        edge_densities = [f['edge_density'] for f in shape_features]
        foreground_ratios = [f['foreground_ratio'] for f in shape_features]
        foreground_counts = [f['foreground_pixel_count'] for f in shape_features]
        circularities = [f['circularity'] for f in shape_features]
        solidities = [f['solidity'] for f in shape_features]
        eccentricities = [f['eccentricity'] for f in shape_features]
        orientations = [f['orientation'] for f in shape_features]
        entropies = [f['entropy'] for f in shape_features]
        region_areas = [f['region_area'] for f in shape_features]
        
        return {
            'mean_intensity': float(np.mean(intensities)),
            'std_intensity': float(np.std(intensities)),
            'intensity_stable': float(np.std(intensities)) < 10,
            'mean_gradient': float(np.mean(gradients)),
            'std_gradient': float(np.std(gradients)),
            'mean_edge_density': float(np.mean(edge_densities)),
            'std_edge_density': float(np.std(edge_densities)),
            'mean_entropy': float(np.mean(entropies)),
            'mean_foreground_pixels': float(np.mean(foreground_counts)),
            'std_foreground_pixels': float(np.std(foreground_counts)),
            'min_foreground_pixels': float(np.min(foreground_counts)),
            'max_foreground_pixels': float(np.max(foreground_counts)),
            'foreground_variance': float(np.var(foreground_counts)),
            'mean_foreground_ratio': float(np.mean(foreground_ratios)),
            'mean_circularity': float(np.mean(circularities)),
            'std_circularity': float(np.std(circularities)),
            'circularity_stable': float(np.std(circularities)) < 0.1,
            'mean_solidity': float(np.mean(solidities)),
            'std_solidity': float(np.std(solidities)),
            'shape_stable': float(np.std(solidities)) < 0.05,
            'mean_eccentricity': float(np.mean(eccentricities)),
            'mean_orientation': float(np.mean(orientations)),
            'orientation_changes': int(np.sum(np.abs(np.diff(orientations)) > 15)),
            'mean_region_area': float(np.mean(region_areas)),
            'relative_size_change': float((region_areas[-1] - region_areas[0]) / region_areas[0]) if region_areas[0] > 0 else 0,
            'relative_foreground_change': float((foreground_counts[-1] - foreground_counts[0]) / foreground_counts[0]) if foreground_counts[0] > 0 else 0,
        }
