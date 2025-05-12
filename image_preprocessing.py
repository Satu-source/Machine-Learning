import os
import shutil
from PIL import Image
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Configuration - modify these as needed
INPUT_PATH = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Data for project\Food_data\combined"
OUTPUT_PATH = r"C:\Users\satuh\OneDrive - Hämeen ammattikorkeakoulu (1)\2025 opinnot\Hamk\Machine learning\Assignment 4\Data for project\processed_food_data"
TARGET_SIZE = (224, 224)  # Standard size for many ML models
QUALITY = 85              # Image quality (1-100)

# Define healthy and unhealthy categories
HEALTHY_CATEGORIES = {
    'Apple', 'Banana', 'Dairy product', 'edamame', 'Egg',
    'lettuce', 'Meat', 'tomato', 'Vegetable-Fruit', 'Bread'
}

UNHEALTHY_CATEGORIES = {
    'Cake', 'Dessert', 'Fried food', 'hamburger',
    'Noodles-Pasta', 'pizza', 'Rice', 'French fries',
    'Hot Dog', 'tacos'
}

def process_image(args):
    """Process and classify a single image"""
    src_path, healthy = args
    
    try:
        # Determine output path
        category = os.path.basename(os.path.dirname(src_path))
        filename = os.path.basename(src_path)
        dest_folder = os.path.join(OUTPUT_PATH, 'healthy' if healthy else 'unhealthy')
        dest_path = os.path.join(dest_folder, f"{category}_{filename}")
        
        # Create destination folder if needed
        os.makedirs(dest_folder, exist_ok=True)
        
        # Process image
        with Image.open(src_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize with aspect ratio preservation
            img.thumbnail((TARGET_SIZE[0]*2, TARGET_SIZE[1]*2), Image.LANCZOS)
            
            # Center crop
            img = np.array(img)
            height, width = img.shape[0], img.shape[1]
            start_x = width//2 - TARGET_SIZE[0]//2
            start_y = height//2 - TARGET_SIZE[1]//2
            img = img[start_y:start_y+TARGET_SIZE[1], start_x:start_x+TARGET_SIZE[0]]
            
            # Save as JPEG with consistent quality
            Image.fromarray(img).save(dest_path, 'JPEG', quality=QUALITY)
        
        return (True, src_path)
    
    except Exception as e:
        return (False, f"{src_path} - {str(e)}")

def classify_and_process():
    """Main function to classify and process all images"""
    # Prepare list of all images with their classification
    tasks = []
    
    for category in os.listdir(INPUT_PATH):
        category_path = os.path.join(INPUT_PATH, category)
        if not os.path.isdir(category_path):
            continue
        
        # Determine classification
        if category in HEALTHY_CATEGORIES:
            healthy = True
        elif category in UNHEALTHY_CATEGORIES:
            healthy = False
        else:
            print(f"Warning: Unclassified category '{category}' - skipping")
            continue
        
        # Add all images in this category to processing queue
        for img_file in os.listdir(category_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(category_path, img_file)
                tasks.append((img_path, healthy))
    
    # Process images in parallel
    print(f"Found {len(tasks)} images to process")
    print(f"Healthy categories: {HEALTHY_CATEGORIES}")
    print(f"Unhealthy categories: {UNHEALTHY_CATEGORIES}")
    
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_PATH, 'healthy'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'unhealthy'), exist_ok=True)
    
    # Process with multiple cores (leave 1 core free)
    num_workers = max(1, cpu_count() - 1)
    success_count = 0
    
    print(f"\nStarting processing with {num_workers} workers...")
    with Pool(num_workers) as pool, tqdm(total=len(tasks)) as pbar:
        for result in pool.imap_unordered(process_image, tasks):
            if result[0]:
                success_count += 1
            else:
                print(f"Error: {result[1]}")
            pbar.update()
    
    print(f"\nProcessing complete! Successfully processed {success_count}/{len(tasks)} images.")

if __name__ == "__main__":
    classify_and_process()