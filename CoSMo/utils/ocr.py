from openai import OpenAI, AsyncOpenAI
from PIL import Image
import os
from io import BytesIO
import base64
import time
import asyncio
from tqdm.asyncio import tqdm
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

data_dir = "/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images"
books = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

image_paths = []
for book in books:
    image_dir = os.path.join(data_dir, book)
    image_files = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    image_paths.extend(image_files)

print(f"{len(books)} books and {len(image_paths)} images loaded for OCR.")

model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

async_client = AsyncOpenAI(
  base_url="http://158.109.8.151:8000/v1",
  api_key="art",
)

prompt = """You are a specialized Vision-Language model with expertise in comic book page analysis and Optical Character Recognition (OCR). Your task is to process an input image of a comic book page and return a structured JSON output. Follow these steps precisely:

1.  **Overall Page Analysis:**
    * Generate a concise description of the page, encompassing its key visual elements (e.g., art style, panel layout complexity, color scheme) and a brief summary of the depicted scene or content. This will populate the "**Description**" field in the JSON output.

2.  **Determine Page Type:**
    * Classify the image into one of the following categories: "**Cover**", "**Story-Start**", "**Story-Page**", "**Advertisement**", or "**Text-Story**".
        * "**Cover**" refers to the **very first page of a comic book**, typically characterized by **prominent graphics, a large main title, and often includes information like issue number, publication date, and creators**. It serves as the primary visual introduction to the comic.
        * "**Story-Start**" refers to the initial page of a narrative story within the comic book; it typically introduces the story and **must contain a story title**.
        * "**Story-Page**" refers to subsequent pages of that narrative.
        * "**Text-Story**" is a page primarily composed of prose narrative, potentially with minor illustrations, distinct from panel-driven comic stories.
        * "**Advertisement**" pages are primarily promotional, even if text-heavy. Differentiate from "Text-Story" by analyzing textual features for promotional language versus narrative prose.
    * Output the classification to the "**PageType**" field in the JSON output.

3.  **Perform OCR and Structure Output based on Page Type:**
    * The results of this step will populate the "**OCRResult**" field in the JSON output. **The "OCRResult" field must contain a single, multi-line string.** The content and structure of this string depend on the "**PageType**". Use the exact labels (e.g., "Title:", "Panel 1:", "AdditionalContent:", "Content:") as shown below, followed by the extracted text.

    * **If "PageType" is "Cover":**
        * Structure the string as:
            Title: <main title text>
            AdditionalContent: <all other prominent text like subtitles, taglines, publisher, creators, price, issue number, etc.>

    * **If "PageType" is "Story-Start":**
        * Structure the string as:
            Title: <story title text present on the page>
            Panel 1: <all text in panel 1 in reading order>
            Panel 2: <all text in panel 2 in reading order>

        * Identify distinct panels based on visual boundaries (frames, gutters, or clear separation, considering various layouts). Determine the correct panel reading order for the page. Transcribe all text (dialogue, narration/captions, sound effects) within each panel in its internal reading order.

    * **If "PageType" is "Story-Page":**
        * Structure the string as:
            Panel 1: <all text in panel 1 in reading order>
            Panel 2: <all text in panel 2 in reading order>

        * Identify distinct panels and transcribe text as described for "**Story-Start**".

    * **If "PageType" is "Advertisement" or "Text-Story":**
        * Structure the string as:
            Content: <all significant blocks of text visible on the page>

4.  **Strict JSON Output Format:**
    * Return *only* a single, valid JSON object. Do not include any introductory text, explanations, or apologies before or after the JSON.
    * **Crucially, the JSON output must not contain any markdown formatting (e.g., backticks, asterisks, hash signs).**
    * The JSON structure must be:

        {
          "Description": "string",
          "PageType": "string (Cover | Story-Start | Story-Page | Advertisement | Text-Story)",
          "OCRResult": "string" // This will be a multi-line string as detailed in step 3
        }

    * Example of "OCRResult" string for a "**Story-Start**":
        "Title: THE ADVENTURES OF CAPTAIN COMET\nPanel 1: IT'S A BIRD! IT'S A PLANE! NO...\nPanel 2: MEANWHILE, AT THE HALL OF JUSTICE..."
    * Ensure all field names in the JSON structure (e.g., "Description", "PageType", "OCRResult") are exactly as written.
    * If a panel has no text, represent it as: Panel <number>: \n (i.e., the label followed by a newline).
"""

async def send_message(message, model_name):
    response = await async_client.chat.completions.create(
                            messages=message['msg'],
                            model=model_name,
                            max_completion_tokens=2048,
                            temperature=0.0,
                        )
    return response, message['img_path']

async def process_batch(image_batch, batch_num, total_batches):
    print(f"\nProcessing batch {batch_num}/{total_batches} ({len(image_batch)} images)")
    
    images_base64 = []
    for img_pth in image_batch:
        try:
            images_base64.append({
                'base64': f"data:image/jpeg;base64,{encode_pil_image(img_pth, resize=1024)}",
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
            {"role": "system", "content": "You are a helpful assistant."},
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
    
    batch_results = []
    batch_errors = []
    
    with tqdm(total=len(tasks), desc=f"Batch {batch_num}", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        for completed_task in asyncio.as_completed(tasks):
            try:
                response, img_path = await completed_task
                ocr_result = response.choices[0].message.content
                batch_results.append((img_path, ocr_result))
 
                pbar.set_postfix_str(f"✓ {os.path.basename(img_path)[:25]}")
                pbar.update(1)
                clean_response = clean_json_response(ocr_result)
                base_name = os.path.splitext(img_path)[0]
                txt_file = f"{base_name}.txt"
                with open(txt_file, "w") as f:
                    f.write(clean_response)

            except Exception as e:
                batch_errors.append((img_path, str(e)))
                pbar.set_postfix_str(f"✗ Error: {str(e)[:20]}")
                pbar.update(1)
    
    return batch_results, batch_errors

async def main():
    start_time = time.time()
    
    BATCH_SIZE = 25 
    MAX_IMAGES = None  
    
    images_to_process = image_paths if MAX_IMAGES is None else image_paths[:MAX_IMAGES]
    
    print(f"Processing {len(images_to_process)} images in batches of {BATCH_SIZE}")
    
    batches = [images_to_process[i:i + BATCH_SIZE] for i in range(0, len(images_to_process), BATCH_SIZE)]
    total_batches = len(batches)
    
    all_results = []
    all_errors = []
    
    for i, batch in enumerate(batches, 1):
        try:
            batch_results, batch_errors = await process_batch(batch, i, total_batches)
            all_results.extend(batch_results)
            all_errors.extend(batch_errors)
            
            print(f"Batch {i} completed: {len(batch_results)} success, {len(batch_errors)} errors")
            print(f"Total progress: {len(all_results)}/{len(images_to_process)} images processed")
            
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue
    
    duration = time.time() - start_time
    print(f"\nAll processing completed in {duration:.2f} seconds.")
    print(f"Total: {len(all_results)} success, {len(all_errors)} errors")
    
    if all_errors:
        print("\nErrors summary:")
        for img_path, error in all_errors[:10]: 
            print(f"  {os.path.basename(img_path)}: {error}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more errors")
    
    return all_results

if __name__ == '__main__': 
    results = asyncio.run(main())