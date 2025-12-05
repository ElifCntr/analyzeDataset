import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Define the paths and initialize statistics containers
dataset_paths = ['/home/elif/Desktop/CVAT/Project1', '/home/elif/Desktop/CVAT/Project2']
drone_count = 0
bird_count = 0
total_image_count = 0
images_with_no_annotations = 0
resolutions = defaultdict(int)
frames_per_video = defaultdict(int)
frames_with_multiple_drones = defaultdict(int)
bounding_box_sizes_drones = []
bounding_box_sizes_birds = []

# Classification of bounding box sizes for drones and birds
size_categories_drones = {
    'small': 0,
    'medium': 0,
    'large': 0
}

size_categories_birds = {
    'small': 0,
    'medium': 0,
    'large': 0
}


# Function to parse YOLO annotation file
def parse_yolo_annotation(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            annotations.append((class_id, x_center, y_center, width, height))
    return annotations


# Traverse through the directories
for dataset_path in dataset_paths:
    directory_count = 0
    for root, dirs, files in os.walk(dataset_path):
        for dir_name in dirs:
            obj_train_data_path = os.path.join(root, dir_name, 'obj_train_data')
            if not os.path.exists(obj_train_data_path):
                continue

            frame_count = 0
            for file_name in os.listdir(obj_train_data_path):
                if file_name.endswith('.txt'):
                    txt_file_path = os.path.join(obj_train_data_path, file_name)
                    img_file_path = txt_file_path.replace('.txt', '.PNG')

                    # Debugging statement to check the paths
                    print(f"Checking image path: {img_file_path}")

                    if not os.path.exists(img_file_path):
                        print(f"Image not found: {img_file_path}")
                        continue

                    # Increment total image count
                    total_image_count += 1

                    # Read image to get the resolution
                    img = cv2.imread(img_file_path)
                    height, width, _ = img.shape
                    resolutions[(width, height)] += 1

                    # Parse annotations
                    annotations = parse_yolo_annotation(txt_file_path)
                    frame_count += 1

                    if len(annotations) == 0:
                        images_with_no_annotations += 1

                    drone_annotation_count = 0

                    for annotation in annotations:
                        class_id, _, _, box_width, box_height = annotation
                        box_width_absolute = box_width * width  # Convert width to absolute values
                        box_height_absolute = box_height * height  # Convert height to absolute values
                        box_area = box_width_absolute * box_height_absolute  # Calculate the actual bounding box area

                        # Classify the bounding box sizes
                        if class_id == 0:  # Assuming 0 is the class_id for drones
                            drone_count += 1
                            drone_annotation_count += 1
                            bounding_box_sizes_drones.append(box_area)
                            if box_area < 32 * 32:
                                size_categories_drones['small'] += 1
                            elif 32 * 32 <= box_area <= 96 * 96:
                                size_categories_drones['medium'] += 1
                            else:
                                size_categories_drones['large'] += 1
                        elif class_id == 1:  # Assuming 1 is the class_id for birds
                            bird_count += 1
                            bounding_box_sizes_birds.append(box_area)
                            if box_area < 32 * 32:
                                size_categories_birds['small'] += 1
                            elif 32 * 32 <= box_area <= 96 * 96:
                                size_categories_birds['medium'] += 1
                            else:
                                size_categories_birds['large'] += 1

                    if drone_annotation_count > 1:
                        frames_with_multiple_drones[dir_name] += 1

            frames_per_video[dir_name] = frame_count



# Plot histograms in separate windows
def plot_size_categories(data, title, xlabel, ylabel, color):
    plt.figure()
    plt.bar(data.keys(), data.values(), color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_histogram(data, title, xlabel, ylabel, color):
    plt.figure()
    plt.hist(data, bins=20, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


plot_size_categories(size_categories_drones, 'Drone Size Categories', 'Category', 'Count', 'green')
plot_size_categories(size_categories_birds, 'Bird Size Categories', 'Category', 'Count', 'blue')
plot_histogram(list(frames_per_video.values()), 'Number of Frames per Video', 'Frames', 'Count', 'blue')
plot_histogram(list(frames_with_multiple_drones.values()), 'Frames with Multiple Drones', 'Frames', 'Count', 'red')
plot_histogram(bounding_box_sizes_drones, 'Bounding Box Sizes of Drones', 'Bounding Box Area', 'Count', 'green')
plot_histogram(bounding_box_sizes_birds, 'Bounding Box Sizes of Birds', 'Bounding Box Area', 'Count', 'purple')

# Print statistics
print(f'Total drone annotations: {drone_count}')
print(f'Total bird annotations: {bird_count}')
print(f'Total image count: {total_image_count}')
print(f'Images with no annotations: {images_with_no_annotations}')
print(f'Resolution frequencies: {dict(resolutions)}')
print(f'Frames per video: {dict(frames_per_video)}')
print(f'Frames with multiple drones: {dict(frames_with_multiple_drones)}')
