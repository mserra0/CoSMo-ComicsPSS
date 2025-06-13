from openai import AsyncOpenAI
from PIL import Image
import os
from io import BytesIO
import base64
import time
import asyncio
from tqdm.asyncio import tqdm
import json
import re

def encode_pil_image(local_url, resize=None):
    image = Image.open(local_url).convert("RGB")
    if resize is not None:
        image_size = image.size
        scale = resize / max(image_size)
        image_size = (int(image_size[0] * scale), int(image_size[1] * scale))
        image = image.resize(image_size)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def clean_json_response(response_text):
    cleaned = re.sub(r'^```json\s*\n?', '', response_text.strip())
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    return cleaned.strip()

def load_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    return json_data

def corrupted_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except Exception:
        return True

output_dir = '/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data'
data_dir = "/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images"
test_path = "/home-local/mserrao/PSSComics/multimodal-comic-pss/EncoderClassifier/data/comics_test.json"

test_json = load_json(test_path)
test_books = [b['hash_code'] for b in test_json]

data = {}
for book in test_books:
    image_dir = os.path.join(data_dir, book)
    if not os.path.isdir(image_dir):
        print(f"Warning: Directory not found for book {book}: {image_dir}")
        data[book] = []
        continue
    
    image_files = []
    for f_name in os.listdir(image_dir):
        full_path = os.path.join(image_dir, f_name)
        if os.path.isfile(full_path) and f_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            if not corrupted_image(full_path):
                image_files.append(full_path)
            else:
                print(f"Skipping corrupted image: {full_path}")
                
    data[book] = sorted(image_files)

print(f"{len(test_books)} books loaded for Zero-shot classification.")

model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

async_client = AsyncOpenAI(
  base_url="http://158.109.8.151:8000/v1",
  api_key="art",
)

prompt = """
### Role: Comic Book Page Classifier (Multimodal Analysis)

### Objective:
Classify individual comic book page images into one of five predefined categories. Each page will be analyzed as a standalone entity, leveraging both visual elements and OCR-extracted text. 

### Categories:
1.  **'cover'**:
    * **Visual Characteristics**: Prominent title/logo, issue number, price, publisher information, dominant artistic imagery, often designed for standalone appeal.
    * **Textual Characteristics**: Typically lacks internal story narrative; contains publication details.
2.  **'first-page'**:
    * **Visual Characteristics**: Establishes the setting or characters of a new story segment; may include significant artwork.
    * **Textual Characteristics**: **Must contain a story title (of any size)**, often includes creator credits, and marks the beginning of a narrative sequence, so there's no preceding dialogue or panels.
3.  **'story'**:
    * **Visual Characteristics**: Consists of sequential panel layouts, speech bubbles, and/or captions that advance the narrative.
    * **Textual Characteristics**: Predominantly dialogue and narration directly related to the ongoing story; lacks introductory titles or credits.
4.  **'text-story'**:
    * **Visual Characteristics**: Characterized by a high text-to-image ratio; minimal artwork (typically 1-2 illustrations), often presented in a newspaper-like column format.
    * **Textual Characteristics**: Paragraph blocks constitute over 70% of the content, resembling prose rather than panel-based dialogue.
5.  **'advertisement'**:
    * **Visual Characteristics**: Designed to promote a product or service; may feature specific product imagery, logos, and calls to action.
    * **Textual Characteristics**: Contains pricing information, promotional language (e.g., "buy now," "free offer"), contact details, or product descriptions.

### Instructions for Analysis:

1.  **Multimodal Examination**:
    * **Visual Analysis**: Scrutinize the page for overall layout, image-to-text ratio, panel structures, graphical elements (e.g., logos, borders), and artistic style.
    * **Textual Analysis (OCR Content)**: Evaluate the OCR-extracted text for the presence and density of titles, creator credits, paragraph blocks, prices, promotional phrases, and dialogue patterns.
2.  **Decision Protocol**:
    * Integrate insights from both visual and textual analyses to determine the most fitting category.
    * If visual and textual signals present conflicting evidence, explicitly acknowledge this uncertainty in the reasoning but still provide a final classification.
    * **Always return a classification for every input page.**

### Output Requirements:

* The output must be a strict JSON object.
* The `confidence_reasoning` field must be a detailed qualitative explanation, explicitly referencing:
    * At least two distinct visual features observed on the page.
    * At least two distinct textual patterns identified from the OCR content.
    * A clear justification for why the *rejected* categories do not apply to the given page.
* Do not include numerical confidence scores.

### Output Template:

  {
    "confidence_reasoning": "Explain your classification based on visual and textual evidence, and why other categories were ruled out.",
    "chosen_label": "one of: 'cover', 'first-page', 'story', 'text-story', 'advertisement'"
  }


### Prohibitions:

    * Do not use markdown formatting or text wrapping within the JSON output.
    * Provide no explanations or commentary outside of the specified JSON structure.
    * Do not include any numerical confidence scores; rely solely on qualitative reasoning.
"""

async def send_message(message, model_name):
    response = await async_client.chat.completions.create(
                            messages=message['msg'],
                            model=model_name,
                            max_completion_tokens=2048,
                            temperature=0.0,
                        )
    return response, message['img_path']

async def process_book(book_id, book_images, book_idx, total_books, json_file):
    print(f"\nProcessing book {book_idx+1}/{total_books} ({len(book_images)} images)")
    
    images_base64 = []
    for img_pth in book_images:
        try:
            images_base64.append({
                'base64': f"data:image/jpeg;base64,{encode_pil_image(img_pth)}",
                'image_path': img_pth
            })
        except Exception as e:
            print(f"Error encoding {img_pth}: {e}")
            continue
    
    messages = []   
    for img_data in images_base64:
        img_b64 = img_data['base64']
        path = img_data['image_path']
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {'url': img_b64}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        messages.append({'msg': msg, 'img_path': path})
    
    tasks = []
    for msg in messages:
        task = send_message(message=msg, model_name=model_name)  
        tasks.append(task)
    
    book_results = []
    book_errors = []
    
    with tqdm(total=len(tasks), desc=f"Book {book_idx+1}", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for completed_task in asyncio.as_completed(tasks):
            try:
                response, img_path = await completed_task
                ocr_result = response.choices[0].message.content
                book_results.append((img_path, ocr_result))
 
                pbar.set_postfix_str(f"✓ {os.path.basename(img_path)[:25]}")
                pbar.update(1)
                
                image_name = os.path.splitext(os.path.basename(img_path))[0]  
                cleaned_result = clean_json_response(ocr_result)  
                json_file[book_id][image_name] = json.loads(cleaned_result)  

            except Exception as e:
                book_errors.append((f'{book_id}/{img_path}', str(e)))
                pbar.set_postfix_str(f"✗ Error: {str(e)[:20]}")
                pbar.update(1)
    
    return book_results, book_errors

async def main():
    start_time = time.time()

    MAX_BOOKS = None  
    total_books = len(data) if MAX_BOOKS is None else MAX_BOOKS
    total_images = sum(len(images) for book, images in data.items())
    print(f"Processing {total_books} books with a total of {total_images} images...")
    
    all_results = []
    all_errors = []
    
    json_file = {}
    
    with tqdm(total=total_books, desc="Processing Books", unit="book",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as main_pbar:
        
        for i, (book_id, book_images) in enumerate(data.items()):
            try:
                json_file[book_id] = {}
                book_results, book_errors = await process_book(book_id, book_images, i, total_books, json_file)
                all_results.extend(book_results)
                all_errors.extend(book_errors)
                
                main_pbar.set_postfix_str(f"Book: {book_id}... ({len(book_results)} success, {len(book_errors)} errors)")
                main_pbar.update(1)
                
                print(f"Book {i+1} completed: {len(book_results)} success, {len(book_errors)} errors")
                print(f"Total progress: {len(all_results)}/{total_images} images processed")
                
            except Exception as e:
                print(f"Error processing book {i+1}: {e}")
                continue
        
    duration = time.time() - start_time
    print(f"\nAll processing completed in {duration:.2f} seconds.")
    print(f"Total: {len(all_results)} success, {len(all_errors)} errors")
    
    output_file = f"{output_dir}/zero-shot_qwen.json"
    with open(output_file, 'w') as f:
        json.dump(json_file, f, indent=2)
    print(f"Results saved to {output_file}")
    
    errors_file = 'errors_log.txt'
    if all_errors:
        print("\nErrors summary:")
        for img_path, error in all_errors[:10]: 
            print(f"  {os.path.basename(img_path)}: {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")
            print(f'Errors saved in {errors_file}')
        with open(errors_file, 'w') as f:
            for img_path, error in all_errors:
                f.write(f"{os.path.basename(img_path)}: {error}\n")
        
    return all_results

if __name__ == '__main__': 
    results = asyncio.run(main())