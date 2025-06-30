import os
import shutil
from PIL import Image
import argparse
from tqdm import tqdm

def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        return False

def clean_book_folder(book_path, output_path, file_ext='.jpg'):

    os.makedirs(output_path, exist_ok=True)
    
    valid_images = []
    for filename in os.listdir(book_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(book_path, filename)
            if is_valid_image(full_path):
                valid_images.append(full_path)
            else:
                print(f"Skipping corrupted image: {full_path}")
    
    def get_page_number(filename):
        try:
            num = ''.join(filter(str.isdigit, os.path.basename(filename)))
            return int(num) if num else 0
        except:
            return 0
    
    valid_images.sort(key=get_page_number)

    for i, src_path in enumerate(valid_images):
        new_filename = f"{i:03d}{file_ext}"
        dst_path = os.path.join(output_path, new_filename)
        
        shutil.copy2(src_path, dst_path)
    
    return len(valid_images)

def process_all_books(root_dir, output_root, file_ext='.jpg'):

    clean_books_dir = os.path.join(output_root, "clean_books")
    os.makedirs(clean_books_dir, exist_ok=True)
    
    book_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"Found {len(book_dirs)} book directories to process")

    total_valid = 0
    total_books_processed = 0
    
    for book_dir in tqdm(book_dirs, desc="Processing books"):
        book_path = os.path.join(root_dir, book_dir)
        output_path = os.path.join(clean_books_dir, book_dir)
        
        num_valid = clean_book_folder(book_path, output_path, file_ext)
        
        if num_valid > 0:
            total_valid += num_valid
            total_books_processed += 1
    
    print(f"Processing complete!")
    print(f"Processed {total_books_processed} books")
    print(f"Saved {total_valid} valid images")
    print(f"Clean books saved to: {clean_books_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean and reorganize comic book images')
    parser.add_argument('--input', required=True, help='Root directory containing book folders')
    parser.add_argument('--output', default='.', help='Output root directory (clean_books will be created here)')
    parser.add_argument('--ext', default='.jpg', help='File extension for output images (.jpg, .png, etc)')
    
    args = parser.parse_args()
    
    process_all_books(args.input, args.output, args.ext)