"""
Modular Excel exporter for dataset analysis results
"""
import pandas as pd
from typing import List, Dict, Optional


class DatasetExcelExporter:
    """Export dataset analysis to Excel with multiple sheets."""
    
    def __init__(self, video_stats: List[Dict], class_names: Dict[int, str]):
        """
        Args:
            video_stats: List of video statistics dicts
            class_names: Dict mapping class_id to name
        """
        self.video_stats = video_stats
        self.class_names = class_names
        self.df = None
    
    def prepare_dataframe(self) -> pd.DataFrame:
        """Prepare main DataFrame from video stats. Override for custom columns."""
        data = []
        
        for video in self.video_stats:
            row = {
                'Video Name': video['video_name'],
                'Resolution': video['resolution'],
                'Total Frames': video['total_frames'],
                'Annotated Frames': video.get('annotated_frames', 0),
                'Empty Frames': video.get('empty_frames', 0),
            }
            
            # Add per-class columns
            for class_id, class_name in self.class_names.items():
                row[f'{class_name.title()} Count'] = video.get(f'class_{class_id}_count', 0)
                row[f'{class_name.title()} Mean Size'] = round(
                    video.get(f'class_{class_id}_mean_size', 0), 1
                )
                
                total = video.get(f'class_{class_id}_count', 0)
                small = video.get(f'class_{class_id}_small_count', 0)
                if total > 0:
                    row[f'{class_name.title()} Small %'] = round(small / total * 100, 1)
                else:
                    row[f'{class_name.title()} Small %'] = 0
            
            # Add custom fields if present
            for key in ['capture_method', 'capture_confidence', 'content_category',
                       'mixed_frames', 'drone_pattern']:
                if key in video:
                    row[key.replace('_', ' ').title()] = video[key]
            
            data.append(row)
        
        self.df = pd.DataFrame(data).sort_values('Video Name')
        return self.df
    
    def create_summary_sheet(self, group_by: str) -> pd.DataFrame:
        """Create summary sheet grouped by a column."""
        if self.df is None:
            self.prepare_dataframe()
        
        # Dynamically aggregate numeric columns
        agg_dict = {
            'Video Name': 'count',
            'Total Frames': 'sum',
        }
        
        # Add class count columns
        for class_name in self.class_names.values():
            col = f'{class_name.title()} Count'
            if col in self.df.columns:
                agg_dict[col] = 'sum'
        
        summary = self.df.groupby(group_by).agg(agg_dict)
        summary = summary.rename(columns={'Video Name': 'Video Count'})
        
        return summary
    
    def export(self, output_file: str, custom_sheets: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Export to Excel with multiple sheets.
        
        Args:
            output_file: Output Excel file path
            custom_sheets: Dict of {sheet_name: DataFrame} for additional sheets
        """
        if self.df is None:
            self.prepare_dataframe()
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main sheet
            self.df.to_excel(writer, sheet_name='Complete Catalog', index=False)
            
            # Add custom sheets if provided
            if custom_sheets:
                for sheet_name, df in custom_sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Excel exported: {output_file}")
        return output_file


class SussexDroneExcelExporter(DatasetExcelExporter):
    """SussexDrone-specific Excel exporter with custom sheets."""
    
    def export(self, output_file: str):
        """Export with SussexDrone-specific sheets."""
        if self.df is None:
            self.prepare_dataframe()
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Complete catalog
            self.df.to_excel(writer, sheet_name='Complete Catalog', index=False)
            
            # Sheet 2-4: Summaries
            if 'Capture Method' in self.df.columns:
                self.create_summary_sheet('Capture Method').to_excel(
                    writer, sheet_name='By Capture Method'
                )
            
            if 'Content Category' in self.df.columns:
                self.create_summary_sheet('Content Category').to_excel(
                    writer, sheet_name='By Content Category'
                )
            
            if 'Drone Pattern' in self.df.columns:
                self.create_summary_sheet('Drone Pattern').to_excel(
                    writer, sheet_name='By Drone Pattern'
                )
            
            # Sheet 5: Videos with birds
            bird_col = 'Bird Count'
            if bird_col in self.df.columns:
                birds = self.df[self.df[bird_col] > 0].copy()
                birds.to_excel(writer, sheet_name='Videos with Birds', index=False)
            
            # Sheet 6: Mixed content
            if 'Mixed Frames' in self.df.columns:
                mixed = self.df[self.df['Mixed Frames'] > 0].copy()
                mixed = mixed.sort_values('Mixed Frames', ascending=False)
                mixed.to_excel(writer, sheet_name='Mixed Content', index=False)
            
            # Sheet 7: Drone-to-drone
            if 'Capture Method' in self.df.columns:
                d2d = self.df[self.df['Capture Method'] == 'drone_to_drone'].copy()
                d2d.to_excel(writer, sheet_name='Drone-to-Drone', index=False)
            
            # Sheet 8: Need verification
            if 'Capture Confidence' in self.df.columns:
                needs_check = self.df[
                    self.df['Capture Confidence'].isin(['low', 'medium', 'none'])
                ].copy()
                needs_check.to_excel(writer, sheet_name='Need Verification', index=False)
        
        print(f"✅ Excel exported: {output_file}")
        print(f"   - 8 sheets created")
        return output_file
