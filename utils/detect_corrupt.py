import os
import csv
from PIL import Image
import argparse
import pandas as pd

def check_corrupted_jpg_images(folder_path):
    """
    Recursively checks a folder and its subfolders for corrupted JPG image files
    and prints their names.
    
    Args:
        folder_path (str): Path to the root folder to scan
    
    Returns:
        list: List of corrupted image filenames with their relative paths
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return []
    
    corrupted_images = []
    jpg_extensions = ['.jpg', '.jpeg']
    total_images = 0
    
    print(f"Scanning {folder_path} and its subfolders for corrupted JPG images...")
    
    # Walk through directory and all subdirectories
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Check if the file has a JPG extension
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in jpg_extensions:
                continue
            
            total_images += 1
            file_path = os.path.join(root, filename)
            # Get relative path for reporting
            rel_path = os.path.relpath(file_path, folder_path)
            
            try:
                # Attempt to open the image
                with Image.open(file_path) as img:
                    # Force image data loading
                    img.verify()
                    # Additional check: try to load the image
                    with Image.open(file_path) as img2:
                        img2.load()
            except Exception as e:
                corrupted_images.append(rel_path)
                print(f"Corrupted: {rel_path} - Error: {str(e)}")
    
    print(f"\nScan complete. Found {len(corrupted_images)} corrupted JPG images out of {total_images} JPG images.")
    return corrupted_images

def clean_csv(splits_path, corrupted_path = 'corrupted.txt'):
    with open(corrupted_path, 'r') as f:
        lines = f.read().splitlines()
    
    remove_set = set(lines)
    
    for filename in os.listdir(splits_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(splits_path, filename)
            
            df = pd.read_csv(file_path)
            
            df['full_path'] = df['book_id'] + '/' + df['page_id'].astype(str).str.zfill(3) + '.jpg'
            filtered_df = df[~df['full_path'].isin(remove_set)]
            filtered_df.drop(columns='full_path', inplace=True)

            filtered_df.to_csv(file_path, index=False)

    print("✅ All CSVs have been modified.")

def main():
    parser = argparse.ArgumentParser(description='Check for corrupted JPG images in a folder and its subfolders')
    parser.add_argument('folder', help='Path to the root folder to scan')
    parser.add_argument('--output', help='Optional output file to write list of corrupted images')
    parser.add_argument('--remove', help='Path to Splits folder to remove the corrupted images')
    
    args = parser.parse_args()
    
    corrupted_list = check_corrupted_jpg_images(args.folder)
    
    if corrupted_list:
        print("\nList of corrupted JPG images:")
        for img in corrupted_list:
            print(img)
        
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    for img in corrupted_list:
                        f.write(f"{img}\n")
                print(f"\nCorrupted image list saved to {args.output}")
            except Exception as e:
                print(f"Error writing to output file: {str(e)}")
                
        if args.remove:
            try:
                clean_csv(args.remove, args.output)
                print(f"\nCorrupted image removed from {args.remove}")
            except Exception as e:
                print(f"Error removing corrupted images: {str(e)}")
    else:
        print("No corrupted JPG images found!")

if __name__ == "__main__":
    main()