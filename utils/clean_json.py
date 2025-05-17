import json 
import argparse 
import os

def clean_json(oringinal_json, out_dir):
    out_json = os.path.join(out_dir, 'clean_new_books.json')
    
    try:
        with open(oringinal_json, 'r', encoding='utf-8') as f:
            books = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {oringinal_json} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: File {oringinal_json} contains invalid JSON")
        return
        
    filtered_books = []
    removed = []
    removed_count = 0
    for book in books:
        if (len(book['textstories']) == 0 and len(book['covers']) == 0 
            and  len(book['advertisements']) == 0):
            removed.append(book['dlid'])
            removed_count += 1
        
        else:
            filtered_books.append(book)
    
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(filtered_books, f, indent=4)
            
    print(f'Removed {removed_count} from the original JSON!')
    print(f'Removed comics by dlid:{removed}')
    print(f'Remaining comics:{len(filtered_books)}. Saved to {out_json}')
    
def merge_comic_json_files(file1, file2, output_file):

    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)

    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    if not (isinstance(data1, list) and isinstance(data2, list)):
        print(f"Error: Both files must contain JSON arrays. Found {type(data1)} and {type(data2)}")
        return
    
    merged_data = data1 + data2
    
    seen_keys = set()
    
    for item in merged_data:
        if 'dlid' not in item:
            continue
            
        key = item['dlid']
        if key not in seen_keys:
            seen_keys.add(key)
        else:
            raise Warning(f'Duplicate in {key}!')

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(merged_data, f_out, indent=4)
    
    print(f"Successfully merged {len(data1)} + {len(data2)} comics into {output_file} with {len(merged_data)} total entries")

    
if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Clean the original JSON file from unannotated books.")
    # parser.add_argument(
    #     "--json_path",
    #     type=str,
    #     required=True,
    #     help="Path to the annotations JSON file."
    # )
    # parser.add_argument(
    #     "--out_dir",
    #     type=str,
    #     required=True,
    #     help="Destination directory for the output JSON."
    # )
    
    # args = parser.parse_args()
    # clean_json(args.json_path, args.out_dir)
    merge_comic_json_files(file1='/home/mserrao/PSSComics/Comics/DatasetDCM/comics100_all/comics_all.json', 
                           file2='/home/mserrao/PSSComics/Comics/datasets.unify/DCM/clean_new_books.json', 
                           output_file='/home/mserrao/PSSComics/Comics/DatasetDCM/comics_all_400.json')