# SussexDrone Dataset Analysis Toolkit

Modular Python toolkit for analyzing YOLO-format drone detection datasets. Provides comprehensive video-level statistics, trajectory analysis, and Excel export capabilities.

## Features

### 📊 **Dataset Analysis**
- Video-level statistics (frame counts, annotations, resolutions)
- Object size categorization (MS-COCO standard: small/medium/large)
- Content categorization (drone-only, bird-only, mixed, background)
- Capture method detection (static ground, dynamic aerial, drone-to-drone)
- Multi-object detection (single vs. multiple drones per frame)

### 🛤️ **Trajectory Analysis**
- IoU-based object tracking across frames
- Motion feature extraction:
  - Trajectory length distribution
  - Average speed and speed variance
  - Hovering detection
  - Direction change frequency
  - Path straightness measurement
- Configurable filtering by capture method

### 📈 **Excel Export**
8-sheet Excel workbook with:
1. Complete video catalog (all attributes)
2. Summary by capture method
3. Summary by content category
4. Summary by drone presence pattern
5. Videos with birds (filtered view)
6. Mixed content videos (co-occurring drones and birds)
7. Drone-to-drone videos (filtered view)
8. Videos needing manual verification

## Installation
```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy>=1.20.0
- Pillow>=8.0.0
- pandas>=1.3.0
- openpyxl>=3.0.0

## Usage

### Basic Dataset Analysis
```bash
python analyze_dataset.py \
    --dataset /path/to/dataset \
    --output my_analysis
```

**Output:** `my_analysis.json` + `my_analysis.xlsx`

### Trajectory Analysis (Static Videos Only)
```bash
python analyze_trajectories.py \
    --dataset /path/to/dataset \
    --output trajectory_results \
    --filter-method static_ground
```

**Output:** `trajectory_results.json` + `trajectory_results.xlsx`

**Filter options:**
- `static_ground` - Ground-based static cameras (default)
- `dynamic_aerial` - Moving aerial cameras
- `drone_to_drone` - Air-to-air drone perspectives

## Project Structure
```
analyzeDataset/
├── dataset_analyzer.py         # Base YOLO dataset analyzer (reusable)
├── sussexdrone_analyzer.py     # SussexDrone-specific extensions
├── excel_exporter.py           # Multi-sheet Excel export module
├── trajectory_analyzer.py      # Trajectory extraction and analysis
├── analyze_dataset.py          # Main runner for dataset analysis
├── analyze_trajectories.py     # Main runner for trajectory analysis
├── example_custom_dataset.py   # Example: adapting for other datasets
└── requirements.txt            # Python dependencies
```

## Module Documentation

### `dataset_analyzer.py`
Base class for YOLO format datasets. Override methods for custom structures:
- `find_videos()` - Locate video folders
- `parse_yolo_annotation()` - Parse annotation files
- `_compute_video_stats()` - Add custom statistics

### `trajectory_analyzer.py`
Reusable trajectory analysis module:
- `extract_trajectories(frame_annotations)` - Track objects across frames
- `compute_trajectory_features(trajectory)` - Compute motion features
- `analyze_trajectories(trajectories)` - Aggregate statistics

### `sussexdrone_analyzer.py`
SussexDrone-specific features:
- Capture method classification (3 perspectives)
- Content categorization (4 types)
- Mixed content detection (novel contribution)
- Drone pattern detection (single/multiple)

## Dataset Structure

Expected structure for SussexDrone:
```
dataset_root/
├── Project1/
│   ├── video_name_1/
│   │   └── obj_train_data/
│   │       ├── frame_000001.PNG
│   │       ├── frame_000001.txt  # YOLO format
│   │       └── ...
│   └── video_name_2/
│       └── obj_train_data/
└── Project2/
    └── ...
```

**YOLO annotation format:** `class_id cx cy w h` (normalized coordinates)

## Example Results

### Dataset Analysis
- **213 videos**, 73,139 frames
- **79,160 drone annotations** (90.2%)
- **8,581 bird annotations** (9.8%)
- **3 capture perspectives:** 62 drone-to-drone, 100 dynamic aerial, 51 static ground
- **17 mixed videos** with 4,514 frames containing both drones and birds

### Trajectory Analysis (Static Videos)
- **~X drone trajectories** extracted
- **Average length:** X frames
- **Hovering percentage:** X%
- **Average straightness:** X.XXX

## Adapting for Other Datasets

See `example_custom_dataset.py` for:
- Custom dataset structures
- Adding new statistics
- Custom Excel sheets

**Basic example:**
```python
from dataset_analyzer import YOLODatasetAnalyzer
from excel_exporter import DatasetExcelExporter

class MyAnalyzer(YOLODatasetAnalyzer):
    def __init__(self, dataset_root):
        class_names = {0: 'car', 1: 'person'}
        super().__init__(dataset_root, class_names)
    
    def find_videos(self, **kwargs):
        # Override for your structure
        return video_folders

analyzer = MyAnalyzer('/path/to/dataset')
results = analyzer.analyze_all()
exporter = DatasetExcelExporter(results, analyzer.class_names)
exporter.export('my_analysis.xlsx')
```

## Key Statistics

### Size Categorization
Uses **MS-COCO standard** (pixel-based):
- **Small:** ≤32×32 pixels (≤1,024 px²)
- **Medium:** 32×32 to 96×96 pixels
- **Large:** >96×96 pixels

### Trajectory Features
- **Length:** Number of frames in trajectory
- **Speed:** Average displacement per frame (normalized coordinates)
- **Hovering:** Speed variance <0.001 AND average speed <0.05
- **Direction changes:** Turns >30 degrees
- **Straightness:** Direct distance / path length (0=curved, 1=straight)

## Citation

If you use this toolkit or the SussexDrone dataset, please cite:
```
[thesis citation will go here]
```

## License

[]

## Author

Elif Canatar  
University of Sussex  
PhD Research - Drone Detection

## Acknowledgments

Supervisor: Dr. Phil Birch  
Department of Engineering and Design  
University of Sussex
