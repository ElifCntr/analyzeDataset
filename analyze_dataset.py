#!/usr/bin/env python3
"""
Main runner for dataset analysis with Excel export
"""
import argparse
import json
from pathlib import Path
from sussexdrone_analyzer import SussexDroneAnalyzer
from excel_exporter import SussexDroneExcelExporter


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset and export to Excel')
    parser.add_argument('--dataset', type=str, required=True, default='/home/elif/Desktop/SussexDrone_dataset',
                       help='Path to dataset root')
    parser.add_argument('--output', type=str, default='analysis',
                       help='Output prefix (without extension)')
    parser.add_argument('--image-folder', type=str, default='obj_train_data',
                       help='Name of folder containing images in each video folder')
    
    args = parser.parse_args()
    
    print("🔍 Starting analysis...")
    
    # Analyze dataset
    analyzer = SussexDroneAnalyzer(args.dataset)
    video_stats = analyzer.analyze_all(args.image_folder)
    
    if not video_stats:
        print("❌ No videos found!")
        return
    
    # Print summary
    overall = analyzer.get_overall_stats()
    print(f"\n✅ Analysis complete!")
    print(f"   Videos: {overall['total_videos']}")
    print(f"   Frames: {overall['total_frames']:,}")
    for class_id, class_name in analyzer.class_names.items():
        count = overall.get(f'total_class_{class_id}', 0)
        print(f"   {class_name.title()}: {count:,}")
    
    # Save JSON
    json_file = f"{args.output}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'videos': video_stats,
            'overall': overall,
            'class_names': analyzer.class_names
        }, f, indent=2)
    print(f"\n📄 JSON saved: {json_file}")
    
    # Export Excel
    excel_file = f"{args.output}.xlsx"
    exporter = SussexDroneExcelExporter(video_stats, analyzer.class_names)
    exporter.export(excel_file)
    
    print(f"\n✅ Done! Check {excel_file}")


if __name__ == '__main__':
    main()
