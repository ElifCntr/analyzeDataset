#!/usr/bin/env python3
"""
Trajectory Analysis with Shape Features
Analyzes motion + shape for static ground videos only
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sussexdrone_analyzer import SussexDroneAnalyzer
from trajectory_analyzer import TrajectoryAnalyzer
from shape_analyzer import ShapeAnalyzer


def load_video_frames(video_folder, img_folder_name='obj_train_data'):
    """Load frame annotations and get image dimensions."""
    img_folder = video_folder / img_folder_name
    if not img_folder.exists():
        return [], 1920, 1080
    
    images = sorted(list(img_folder.glob('*.PNG')) + list(img_folder.glob('*.png')))
    if not images:
        return [], 1920, 1080
    
    try:
        with Image.open(images[0]) as img:
            img_width, img_height = img.size
    except:
        img_width, img_height = 1920, 1080
    
    frame_annotations = []
    for img_path in images:
        txt_path = img_path.with_suffix('.txt')
        frame_anns = []
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        frame_anns.append({
                            'class_id': int(parts[0]),
                            'cx': float(parts[1]),
                            'cy': float(parts[2]),
                            'w': float(parts[3]),
                            'h': float(parts[4])
                        })
        frame_annotations.append(frame_anns)
    
    return frame_annotations, img_width, img_height


def main():
    parser = argparse.ArgumentParser(description='Trajectory + Shape Analysis')
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--output', default='trajectory_analysis', help='Output prefix')
    parser.add_argument('--filter-method', default='static_ground',
                       help='Capture method to analyze (static_ground/dynamic_aerial/drone_to_drone)')
    parser.add_argument('--analyze-shapes', action='store_true',
                       help='Enable detailed shape analysis (slower)')
    parser.add_argument('--max-trajectories-per-video', type=int, default=3,
                       help='Max trajectories to analyze per video for shape analysis')
    args = parser.parse_args()
    
    print("🔍 Step 1: Categorizing videos...")
    analyzer = SussexDroneAnalyzer(args.dataset)
    video_stats = analyzer.analyze_all()
    
    filtered_videos = [v for v in video_stats 
                      if v.get('capture_method') == args.filter_method]
    
    print(f"\n📊 Dataset breakdown:")
    print(f"   Total videos: {len(video_stats)}")
    print(f"   {args.filter_method}: {len(filtered_videos)}")
    print(f"   EXCLUDED: {len(video_stats) - len(filtered_videos)}")
    
    if not filtered_videos:
        print(f"\n❌ No {args.filter_method} videos found!")
        return
    
    if args.analyze_shapes:
        print(f"\n⚠️  Shape analysis ENABLED - analyzing up to {args.max_trajectories_per_video} trajectories per video")
    
    print(f"\n🛤️  Step 2: Extracting trajectories...")
    
    traj_drone = TrajectoryAnalyzer(class_id=0)
    traj_bird = TrajectoryAnalyzer(class_id=1)
    
    if args.analyze_shapes:
        shape_analyzer = ShapeAnalyzer()
    
    all_drone_traj = []
    all_bird_traj = []
    per_video_results = []
    detailed_shape_data = []
    
    dataset_root = Path(args.dataset)
    
    for v in tqdm(filtered_videos, desc="Processing videos", unit="video"):
        video_name = v['video_name']
        
        video_folder = None
        for project in dataset_root.iterdir():
            if not project.is_dir():
                continue
            candidate = project / video_name
            if candidate.exists():
                video_folder = candidate
                break
        
        if not video_folder:
            continue
        
        frame_anns, img_w, img_h = load_video_frames(video_folder)
        if not frame_anns:
            continue
        
        video_drone_traj = traj_drone.extract_trajectories(frame_anns, img_w, img_h)
        video_bird_traj = traj_bird.extract_trajectories(frame_anns, img_w, img_h)
        
        video_drone_stats = traj_drone.analyze_trajectories(video_drone_traj)
        video_bird_stats = traj_bird.analyze_trajectories(video_bird_traj)
        
        # Shape analysis (optional)
        video_shape_stats = {}
        if args.analyze_shapes and len(video_drone_traj) > 0:
            for traj_idx, traj in enumerate(video_drone_traj[:args.max_trajectories_per_video]):
                shape_features = shape_analyzer.analyze_trajectory_shapes(video_folder, traj)
                if shape_features:
                    shape_stats = shape_analyzer.compute_shape_change_statistics(shape_features)
                    detailed_shape_data.append({
                        'video_name': video_name,
                        'trajectory_id': traj_idx,
                        'trajectory_length': traj['length'],
                        **shape_stats
                    })
                    if traj_idx == 0:
                        video_shape_stats = shape_stats
        
        per_video_results.append({
            'video_name': video_name,
            'total_frames': len(frame_anns),
            'resolution': f"{img_w}x{img_h}",
            'drone_trajectories': video_drone_stats['num_trajectories'],
            'drone_avg_length': video_drone_stats['avg_length'],
            'drone_hovering_pct': video_drone_stats.get('hovering_percentage', 0),
            'drone_avg_speed': video_drone_stats.get('avg_speed', 0),
            'drone_avg_area': video_drone_stats.get('avg_object_area', 0),
            'drone_approaching_pct': video_drone_stats.get('approaching_percentage', 0),
            'drone_stable_shape_pct': video_drone_stats.get('stable_shape_percentage', 0),
            'bird_trajectories': video_bird_stats['num_trajectories'],
            'bird_avg_length': video_bird_stats['avg_length'],
            'mean_foreground_pixels': video_shape_stats.get('mean_foreground_pixels', 0) if args.analyze_shapes else 0,
            'mean_circularity': video_shape_stats.get('mean_circularity', 0) if args.analyze_shapes else 0,
        })
        
        all_drone_traj.extend(video_drone_traj)
        all_bird_traj.extend(video_bird_traj)
    
    print(f"\n📊 Step 3: Computing aggregate statistics...")
    
    drone_stats = traj_drone.analyze_trajectories(all_drone_traj)
    bird_stats = traj_bird.analyze_trajectories(all_bird_traj)
    
    # Shape aggregate
    shape_aggregate = {}
    if args.analyze_shapes and detailed_shape_data:
        foreground_pixels = [d['mean_foreground_pixels'] for d in detailed_shape_data]
        circularities = [d['mean_circularity'] for d in detailed_shape_data]
        solidities = [d['mean_solidity'] for d in detailed_shape_data]
        edge_densities = [d['mean_edge_density'] for d in detailed_shape_data]
        
        shape_aggregate = {
            'trajectories_analyzed': len(detailed_shape_data),
            'mean_foreground_pixels': float(np.mean(foreground_pixels)),
            'mean_circularity': float(np.mean(circularities)),
            'mean_solidity': float(np.mean(solidities)),
            'mean_edge_density': float(np.mean(edge_densities)),
        }
    
    # Print results
    print(f"\n{'='*80}")
    print(f"TRAJECTORY ANALYSIS - {args.filter_method.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    print(f"\n🚁 DRONE TRAJECTORIES ({drone_stats['num_trajectories']} total):")
    print(f"\n   Length: {drone_stats['avg_length']:.1f} avg ({drone_stats['min_length']}-{drone_stats['max_length']} range)")
    print(f"   Distribution: {drone_stats['length_distribution']['short']} short, {drone_stats['length_distribution']['medium']} medium, {drone_stats['length_distribution']['long']} long")
    print(f"\n   Motion: Speed {drone_stats['avg_speed']:.4f}, Hovering {drone_stats['hovering_percentage']:.1f}%, Straightness {drone_stats['avg_straightness']:.3f}")
    print(f"   Appearance: Size {drone_stats['avg_object_area']:.1f} px², Approaching {drone_stats['approaching_percentage']:.1f}%, Stable {drone_stats['stable_shape_percentage']:.1f}%")
    
    if shape_aggregate:
        print(f"\n   Shape Features:")
        print(f"      Foreground pixels: {shape_aggregate['mean_foreground_pixels']:.1f} px")
        print(f"      Circularity: {shape_aggregate['mean_circularity']:.3f}")
        print(f"      Solidity: {shape_aggregate['mean_solidity']:.3f}")
        print(f"      Edge density: {shape_aggregate['mean_edge_density']:.3f}")
    
    print(f"\n🦅 BIRD TRAJECTORIES ({bird_stats['num_trajectories']} total):")
    if bird_stats['num_trajectories'] > 0:
        print(f"   Length: {bird_stats['avg_length']:.1f} avg, Hovering: {bird_stats['hovering_percentage']:.1f}%")
    
    print(f"\n{'='*80}")
    
    # Save
    results = {
        'filter_method': args.filter_method,
        'shape_analysis_enabled': args.analyze_shapes,
        'videos_analyzed': len(filtered_videos),
        'drone_trajectories': drone_stats,
        'bird_trajectories': bird_stats,
        'shape_aggregate': shape_aggregate,
        'per_video': per_video_results,
        'detailed_shape_analysis': detailed_shape_data if args.analyze_shapes else [],
    }
    
    json_file = f"{args.output}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 JSON: {json_file}")
    
    # Excel
    excel_file = f"{args.output}.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Drone stats
        drone_rows = []
        for k, v in drone_stats.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    drone_rows.append({'Metric': f'{k}_{sub_k}', 'Value': sub_v})
            else:
                drone_rows.append({'Metric': k, 'Value': v})
        pd.DataFrame(drone_rows).to_excel(writer, sheet_name='Drone Overall', index=False)
        
        # Bird stats
        bird_rows = []
        for k, v in bird_stats.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    bird_rows.append({'Metric': f'{k}_{sub_k}', 'Value': sub_v})
            else:
                bird_rows.append({'Metric': k, 'Value': v})
        pd.DataFrame(bird_rows).to_excel(writer, sheet_name='Bird Overall', index=False)
        
        # Per video
        per_video_df = pd.DataFrame(per_video_results).sort_values('drone_trajectories', ascending=False)
        per_video_df.to_excel(writer, sheet_name='Per Video', index=False)
        
        # Top 20
        per_video_df.head(20).to_excel(writer, sheet_name='Top 20 Videos', index=False)
        
        # Shape analysis
        if args.analyze_shapes and detailed_shape_data:
            pd.DataFrame(detailed_shape_data).to_excel(writer, sheet_name='Shape Analysis', index=False)
    
    sheet_count = 5 if args.analyze_shapes and detailed_shape_data else 4
    print(f"📊 Excel: {excel_file} ({sheet_count} sheets)")
    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()
