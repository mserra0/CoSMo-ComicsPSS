import json
import os
import re
import sys
from typing import List, Dict, Any, Optional
import argparse

def fix_and_parse_json(json_str: str, file) -> Optional[Dict[str, Any]]:
    fixed_chars = []
    in_string = False
    is_escaped = False

    for char in json_str:
        if char == '"' and not is_escaped:
            in_string = not in_string
        
        is_escaped = (char == '\\' and not is_escaped)

        if in_string and char == '\n':
            fixed_chars.append('\\n')
        elif in_string and char == '\r':
            pass
        else:
            fixed_chars.append(char)
    
    fixed_json_str = "".join(fixed_chars)

    try:
        return json.loads(fixed_json_str)
    except json.JSONDecodeError as e:
        print(f"  - Warning: Failed to parse a block in file {file} even after fixing attempts. Error: {e}")
        return None

def find_and_parse_json_objects(text: str, file) -> List[Dict[str, Any]]:
    potential_json_strings = re.findall(r'\{.*?\}', text, re.DOTALL)
    
    parsed_objects = []
    for json_str in potential_json_strings:
        parsed_obj = fix_and_parse_json(json_str, file)
        if parsed_obj:
            parsed_objects.append(parsed_obj)
            
    return parsed_objects

def choose_correct_json(objects, verbose = True):
    if not objects:
        return None
        
    if len(objects) == 2 and objects[0] == objects[1]:
        if verbose:
            print("  - Found two identical JSON objects. Keeping the second one.")
        return objects[1]

    if len(objects) > 1:
        if verbose:
            print(f"  - Found {len(objects)} JSON objects. Selecting the last valid one as per default logic.")
        
    return objects[-1]

def process_file(filepath: str, verbose=True):
    if verbose:
        print(f"Processing '{os.path.basename(filepath)}'...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        parsed_objects = find_and_parse_json_objects(content, filepath)
        chosen_object = choose_correct_json(parsed_objects, verbose)

        if chosen_object:
            clean_content = json.dumps(chosen_object, indent=2)
            
            base, ext = os.path.splitext(filepath)
            json_file = f'{base}.json'
            
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(clean_content)
            if verbose:
                print(f"  - Cleaned JSON saved to '{os.path.basename(json_file)}'.")
        else:
            print(f"  - No valid JSON found in this file {filepath}. No action taken.")

    except FileNotFoundError:
        print(f"  - Error: File not found at '{filepath}'")
    except Exception as e:
        print(f"  - An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Cleans OCR output files containing JSON objects.")
    parser.add_argument("--target_dir", default=None,
                        help="Directory containing .txt files to process. Defaults to 'ocr_files' in script's directory.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase output verbosity.")
    
    args = parser.parse_args()

    target_dir = args.target_dir
    if target_dir is None:
        print(f"No directory provided. Defaulting to: '{target_dir}'")

    if not os.path.isdir(target_dir):
        print(f"Error: The directory '{target_dir}' does not exist.")
        print("Usage: python your_script_name.py --target_dir /path/to/your/files (optional) --verbose")
        return
    
    num_directories_scanned = 0
    num_files_processed = 0

    for root, dirs, files in os.walk(target_dir):
        num_directories_scanned += 1
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                process_file(file_path, verbose=args.verbose) 
                num_files_processed += 1
                
    print(f'All {num_directories_scanned-1} directories with a total of {num_files_processed} OCR files has been processed!')

if __name__ == "__main__":
    main()