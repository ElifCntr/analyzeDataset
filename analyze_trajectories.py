#!/usr/bin/env python3
"""Analyze trajectories - STATIC GROUND VIDEOS ONLY"""
import argparse
import json
import pandas as pd
from pathlib import Path
from sussexdrone_analyzer import SussexDroneAnalyzer
from trajectory_analyzer import TrajectoryAnalyzer


def load_video_frames(video_folder, img_folder_name='obj_train_data'):
    """Load frame-by-frame annotations from video folder."""
    img_folder = video_folder / img_folder_name
    if not img_folder.exists():
        return []

    images = sorted(list(img_folder.glob('*.PNG')) + list(img_folder.glob('*.png')))
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

    return frame_annotations


def main():
    parser = argparse.ArgumentParser(description='Trajectory analysis for static videos')
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--output', default='trajectory_analysis', help='Output prefix')
    parser.add_argument('--filter-method', default='static_ground',
                        help='Capture method to analyze (static_ground/dynamic_aerial/drone_to_drone)')
    args = parser.parse_args()

    print("🔍 Step 1: Categorizing videos...")
    analyzer = SussexDroneAnalyzer(args.dataset)
    video_stats = analyzer.analyze_all()

    # Filter videos by capture method
    filtered_videos = [v for v in video_stats
                       if v.get('capture_method') == args.filter_method]

    print(f"\n📊 Dataset breakdown:")
    print(f"   Total videos: {len(video_stats)}")
    print(f"   {args.filter_method}: {len(filtered_videos)}")
    print(f"   EXCLUDED: {len(video_stats) - len(filtered_videos)}")

    if not filtered_videos:
        print(f"\n❌ No {args.filter_method} videos found!")
        return

    print(f"\n🛤️  Step 2: Extracting trajectories from {len(filtered_videos)} videos...")

    # Initialize trajectory analyzers
    traj_drone = TrajectoryAnalyzer(class_id=0)  # Drones
    traj_bird = TrajectoryAnalyzer(class_id=1)  # Birds
    all_drone_traj, all_bird_traj = [], []

    # Process each video
    dataset_root = Path(args.dataset)
    for i, v in enumerate(filtered_videos, 1):
        if i % 10 == 0:
            print(f"   Processed {i}/{len(filtered_videos)}...")

        # Find video folder
        video_folder = None
        for project in dataset_root.iterdir():
            if not project.is_dir():
                continue
            candidate = project / v['video_name']
            if candidate.exists():
                video_folder = candidate
                break

        if not video_folder:
            continue

        # Load frame annotations and extract trajectories
        frame_anns = load_video_frames(video_folder)
        if frame_anns:
            all_drone_traj.extend(traj_drone.extract_trajectories(frame_anns))
            all_bird_traj.extend(traj_bird.extract_trajectories(frame_anns))

    print(f"\n📊 Step 3: Computing statistics...")
    drone_stats = traj_drone.analyze_trajectories(all_drone_traj)
    bird_stats = traj_bird.analyze_trajectories(all_bird_traj)

    # Print results
    print(f"\n{'=' * 80}")
    print(f"TRAJECTORY ANALYSIS - {args.filter_method.upper().replace('_', ' ')}")
    print(f"{'=' * 80}")

    print(f"\n🚁 DRONE TRAJECTORIES:")
    print(f"   Total: {drone_stats['num_trajectories']}")
    print(f"   Avg length: {drone_stats['avg_length']:.1f} frames")
    print(f"   Range: {drone_stats['min_length']}-{drone_stats['max_length']} frames")
    print(f"   Distribution: Short={drone_stats['length_distribution']['short']}, "
          f"Medium={drone_stats['length_distribution']['medium']}, "
          f"Long={drone_stats['length_distribution']['long']}")
    print(f"   Avg speed: {drone_stats['avg_speed']:.4f} (normalized)")
    print(f"   Hovering: {drone_stats['hovering_percentage']:.1f}%")
    print(f"   Avg direction changes: {drone_stats['avg_direction_changes']:.1f}")
    print(f"   Avg straightness: {drone_stats['avg_straightness']:.3f}")

    print(f"\n🦅 BIRD TRAJECTORIES:")
    print(f"   Total: {bird_stats['num_trajectories']}")
    if bird_stats['num_trajectories'] > 0:
        print(f"   Avg length: {bird_stats['avg_length']:.1f} frames")
        print(f"   Hovering: {bird_stats['hovering_percentage']:.1f}%")
        print(f"   Avg straightness: {bird_stats['avg_straightness']:.3f}")

    print(f"{'=' * 80}")

    # Save results
    results = {
        'filter_method': args.filter_method,
        'videos_analyzed': len(filtered_videos),
        'videos_excluded': len(video_stats) - len(filtered_videos),
        'drone_trajectories': drone_stats,
        'bird_trajectories': bird_stats,
    }

    json_file = f"{args.output}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 JSON saved: {json_file}")

    # Create Excel
    excel_file = f"{args.output}.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Drone stats
        drone_df = pd.DataFrame([
            {'Metric': k, 'Value': v}
            for k, v in drone_stats.items()
            if not isinstance(v, dict)
        ])
        drone_df.to_excel(writer, sheet_name='Drone Trajectories', index=False)

        # Bird stats
        bird_df = pd.DataFrame([
            {'Metric': k, 'Value': v}
            for k, v in bird_stats.items()
            if not isinstance(v, dict)
        ])
        bird_df.to_excel(writer, sheet_name='Bird Trajectories', index=False)

    print(f"📊 Excel saved: {excel_file}")
    print(f"\n✅ Done!")


if __name__ == '__main__':
    main()


