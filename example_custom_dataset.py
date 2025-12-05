"""
Example: Using the modular analyzer for a different dataset

This shows how to adapt the code for any YOLO format dataset.
"""
from dataset_analyzer import YOLODatasetAnalyzer
from excel_exporter import DatasetExcelExporter


# Example 1: Simple dataset with different structure
class MyDatasetAnalyzer(YOLODatasetAnalyzer):
    """Custom analyzer for your dataset."""
    
    def __init__(self, dataset_root: str):
        # Define your classes
        class_names = {
            0: 'car',
            1: 'person',
            2: 'bicycle'
        }
        
        super().__init__(dataset_root, class_names)
    
    def find_videos(self, **kwargs):
        """
        Override if your dataset structure is different.
        Example: Flat structure with no nested projects
        """
        video_folders = []
        for item in self.dataset_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if this folder has images
                images_folder = item / 'images'  # Your structure
                if images_folder.exists():
                    video_folders.append(item)
        return video_folders
    
    def analyze_video(self, video_folder, **kwargs):
        """
        Override if images are in different location.
        Example: images/ instead of obj_train_data/
        """
        img_folder = video_folder / 'images'  # Your structure
        # ... rest of analysis
        return super().analyze_video(video_folder, image_folder_name='images')


# Example 2: Use it
if __name__ == '__main__':
    # Analyze
    analyzer = MyDatasetAnalyzer('/path/to/your/dataset')
    results = analyzer.analyze_all(image_folder_name='images')
    
    # Export
    exporter = DatasetExcelExporter(results, analyzer.class_names)
    exporter.export('my_dataset_analysis.xlsx')


# Example 3: Custom Excel sheets
class CustomExcelExporter(DatasetExcelExporter):
    """Add your own Excel sheets."""
    
    def export(self, output_file: str):
        if self.df is None:
            self.prepare_dataframe()
        
        # Create custom sheets
        custom_sheets = {}
        
        # Example: Group by resolution
        custom_sheets['By Resolution'] = self.create_summary_sheet('Resolution')
        
        # Example: Filter for specific criteria
        if 'Car Count' in self.df.columns:
            cars_only = self.df[self.df['Car Count'] > 0]
            custom_sheets['Videos with Cars'] = cars_only
        
        # Call parent with custom sheets
        return super().export(output_file, custom_sheets)
