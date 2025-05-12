import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

# Configuration
INPUT_PATH = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Data for project\processed_food_data"
OUTPUT_PATH = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Data for project\augmented_split_dataset"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
SEED = 42
AUGMENTATION_FACTOR = 2  # Generate 2x augmented images for training set only

# ------ ADDED MISSING FUNCTION ------
def create_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
# -----------------------------------

def augment_image(img_path, output_dir, class_name):
    """Generate augmented versions of an image"""
    img = Image.open(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    augmented_images = []
    
    # Horizontal flip
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_images.append((img_flip, f"{base_name}_flip.jpg"))
    
    # Rotation (10 degrees)
    img_rot = img.rotate(10, expand=True)
    img_rot = img_rot.resize(img.size)  # Maintain original size
    augmented_images.append((img_rot, f"{base_name}_rot10.jpg"))
    
    # Save augmented images
    for aug_img, fname in augmented_images:
        aug_img.save(os.path.join(output_dir, class_name, fname))

def process_class(class_name):
    """Process and split a single class"""
    class_path = os.path.join(INPUT_PATH, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Initial split
    train_val, test = train_test_split(images, test_size=TEST_SIZE, random_state=SEED)
    train, val = train_test_split(train_val, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=SEED)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        create_dir(os.path.join(OUTPUT_PATH, split, class_name))
    
    # Copy files
    for split, files in [('train', train), ('val', val), ('test', test)]:
        for file in files:
            src = os.path.join(class_path, file)
            dst = os.path.join(OUTPUT_PATH, split, class_name, file)
            shutil.copy2(src, dst)
            
            # Augment only training set
            if split == 'train':
                augment_image(src, os.path.join(OUTPUT_PATH, split), class_name)
    
    return len(train)*AUGMENTATION_FACTOR, len(val), len(test)

if __name__ == "__main__":
    print("Creating augmented split dataset...")
    create_dir(OUTPUT_PATH)
    
    # Process both classes in parallel
    with Pool(2) as pool:
        results = pool.map(process_class, ['healthy', 'unhealthy'])
    
    # Print summary
    print("\nDataset split complete:")
    print(f"Train set: {results[0][0]+results[1][0]} images (augmented)")
    print(f"Val set:   {results[0][1]+results[1][1]} images")
    print(f"Test set:  {results[0][2]+results[1][2]} images")
    print(f"\nFolder structure created at: {OUTPUT_PATH}")