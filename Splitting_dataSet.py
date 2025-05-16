import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration
INPUT_PATH = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Data for project\processed_food_data"
OUTPUT_PATH = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Data for project\split_dataset"
TEST_SIZE = 0.15  # 15% for test
VAL_SIZE = 0.15   # 15% for validation (remaining 70% for training)
SEED = 42         # Random seed for reproducibility

def create_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset():
    # Create output directories
    for folder in ['train', 'val', 'test']:
        for class_name in ['healthy', 'unhealthy']:
            create_dir(os.path.join(OUTPUT_PATH, folder, class_name))

    # Process each class
    for class_name in ['healthy', 'unhealthy']:
        class_path = os.path.join(INPUT_PATH, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split into train+temp (85%) and test (15%)
        train_val, test = train_test_split(
            images, 
            test_size=TEST_SIZE, 
            random_state=SEED
        )
        
        # Split train_val into train (70%) and val (15%)
        train, val = train_test_split(
            train_val,
            test_size=VAL_SIZE/(1-TEST_SIZE),  # Adjusted split ratio
            random_state=SEED
        )
        
        # Copy files to their respective folders
        for split, files in [('train', train), ('val', val), ('test', test)]:
            for file in files:
                src = os.path.join(class_path, file)
                dst = os.path.join(OUTPUT_PATH, split, class_name, file)
                shutil.copy2(src, dst)
                
        print(f"{class_name}: {len(train)} train, {len(val)} val, {len(test)} test")

if __name__ == "__main__":
    print("Splitting dataset into train/val/test...")
    split_dataset()
    print("\nDone! Folder structure created at:", OUTPUT_PATH)
