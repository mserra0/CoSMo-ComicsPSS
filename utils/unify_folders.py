import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_metadata(json_path: Path) -> Dict[str, str]:
    """Loads the JSON metadata file and creates a dlid to hash_code mapping."""
    dlid_to_hash: Dict[str, str] = {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)
            for item in data:
                dlid = item.get('dlid')
                hash_code = item.get('hash_code')
                if dlid and hash_code:
                    dlid_to_hash[str(dlid)] = str(hash_code)
                else:
                    logging.warning(f"Skipping item due to missing 'dlid' or 'hash_code': {item}")
    except FileNotFoundError:
        logging.error(f"JSON metadata file not found: {json_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {json_path}")
        raise
    return dlid_to_hash

def process_book_folders(
    src_dir: Path,
    dst_dir: Path,
    dlid_to_hash: Dict[str, str],
    dry_run: bool = False
) -> None:
    """Processes book folders, renames, and reorganizes images."""
    if not src_dir.is_dir():
        logging.error(f"Source directory not found or is not a directory: {src_dir}")
        return

    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
    elif not dst_dir.exists():
        logging.info(f"[DRY RUN] Would create destination base directory: {dst_dir}")
    elif not dst_dir.is_dir():
        logging.error(f"Destination path exists but is not a directory: {dst_dir}")
        return
    
    for dlid, hash_code in dlid_to_hash.items():
        dlid_folder = src_dir / dlid

        if not dlid_folder.is_dir():
            logging.warning(f"Source folder for dlid='{dlid}' not found at '{dlid_folder}'. Skipping.")
            continue

        destination_hash_folder = dst_dir / hash_code

        if destination_hash_folder.exists():
            logging.warning(
                f"Destination folder '{destination_hash_folder}' already exists. Skipping dlid='{dlid}' to prevent overwrite."
            )
            continue

        image_files: List[Path] = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*JPG']:
            image_files.extend(sorted(list(dlid_folder.glob(ext))))
        
        if not image_files:
            logging.info(f"No image files found in dlid='{dlid}' folder. Skipping.")
            continue

        if not dry_run:
            try:
                destination_hash_folder.mkdir(parents=True, exist_ok=False)
            except FileExistsError: 
                logging.warning(
                    f"Race condition or unexpected: Destination folder '{destination_hash_folder}' created by another process. Skipping."
                )
                continue
            except OSError as e:
                logging.error(f"Could not create directory {destination_hash_folder}: {e}")
                continue


        logging.info(
            f"{'[DRY RUN] Would process' if dry_run else 'Processing'} dlid='{dlid}' (found {len(image_files)} images) -> hash='{hash_code}'"
        )

        copied_files_count = 0
        for i, old_image_path in enumerate(image_files):
            new_image_name = f"{i:03d}.jpg" 
            new_image_path = destination_hash_folder / new_image_name

            if dry_run:
                logging.info(
                    f"  [DRY RUN] Would copy '{old_image_path.name}' to '{new_image_path}'"
                )
                copied_files_count +=1
            else:
                try:
                    shutil.copy2(old_image_path, new_image_path)
                    copied_files_count += 1
                except Exception as e:
                    logging.error(f"Error copying '{old_image_path}' to '{new_image_path}': {e}")
        
        if copied_files_count > 0 :
             logging.info(
                f"{'[DRY RUN] Would have copied' if dry_run else 'Successfully copied'} {copied_files_count} images for dlid='{dlid}' to '{destination_hash_folder}'."
            )
        elif not dry_run :
             logging.warning(f"No files were copied for dlid='{dlid}'. Check logs for errors.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify and reorganize comic book image folders.")
    parser.add_argument(
        "--json_path",
        type=Path,
        required=True,
        help="Path to the annotations JSON file mapping dlid to hash_code."
    )
    parser.add_argument(
        "--src_dir",
        type=Path,
        default=Path("dcm_22k/images"),
        help="Source directory containing dlid subfolders (default: dcm_22k/images)."
    )
    parser.add_argument(
        "--dst_dir",
        type=Path,
        default=Path("datasets.unify/DCM/images"),
        help="Destination directory for hash_code subfolders (default: datasets.unify/DCM/images)."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, script will only log actions without performing filesystem operations."
    )

    args = parser.parse_args()

    if args.dry_run:
        logging.info("Performing a DRY RUN. No files will be moved or created.")

    try:
        dlid_to_hash_map = load_json_metadata(args.json_path)
        if not dlid_to_hash_map:
            logging.error("Failed to load or parse JSON metadata. Exiting.")
            return

        process_book_folders(args.src_dir, args.dst_dir, dlid_to_hash_map, args.dry_run)
        logging.info("Script execution completed.")

    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
