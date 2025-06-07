import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import random
import json
import argparse
import PIL
import google.generativeai as genai
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()
API_KEYS = os.getenv("API_KEYS2").split(",")

DEFAULT_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

generation_config = {
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

class KeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.key_index = 0
        self.request_counts = {i: 0 for i in range(len(api_keys))}
        self.lock = asyncio.Lock()
        self.config_lock = asyncio.Lock()

    async def get_next_key(self):
        async with self.lock:
            start_index = self.key_index
            while True:
                if self.request_counts[self.key_index] < 1500:
                    self.request_counts[self.key_index] += 1
                    return self.api_keys[self.key_index]
                
                self.key_index = (self.key_index + 1) % len(self.api_keys)
                
                if self.key_index == start_index:
                    raise RuntimeError("All API keys have reached the daily limit.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from images or prompts using Google Generative AI."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to the input JSON file containing keys and prompts.",
    )
    parser.add_argument(
        "--output_file",
        default="category_prompts_answers.jsonl",
        help="File to save results.",
    )
    parser.add_argument(
        "--delay", type=float, default=5.2, help="Delay between processing each prompt."
    )
    parser.add_argument(
        "--sys_instruction", default="", help="System instruction for model."
    )
    parser.add_argument("--temperature", type=float, default=1.5, help="Temperature.")

    return parser.parse_args()

async def get_output(model, prompt, img=None, safety_settings=DEFAULT_SAFETY_SETTINGS):
    try:
        if img is None:
            r = await model.generate_content_async(
                prompt, safety_settings=safety_settings
            )
        else:
            new_img = (
                PIL.Image.open(img).convert("RGB")
                if img.endswith(".gif")
                else PIL.Image.open(img)
            )
            r = await model.generate_content_async(
                [prompt, new_img], safety_settings=safety_settings
            )
        return r.text
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None

async def process_job(key_manager, args, prompt, key, index, results, system_instruction, generation_config):
    await asyncio.sleep(index * args.delay)
    try:
        async with key_manager.config_lock:
            current_api_key = await key_manager.get_next_key()
            genai.configure(api_key=current_api_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction,
                generation_config=generation_config,
            )
        
        result = await get_output(model, prompt)
        if result:
            with open(args.output_file, "a") as f:
                f.write(json.dumps({"key": key, "result": result}) + "\n")
            results.append({"key": key, "result": result})
            print(f"Processed: {key} - {result[:50]}...")
    except Exception as e:
        print(f"Error processing job {index}: {e}")

async def main():
    args = parse_args()
    print(f"Arguments: {args}")

    with open(args.input_file, "r") as f:
        prompts = json.load(f)

    with open(args.sys_instruction, "r") as f:
        system_instruction = f.read().strip()

    generation_config["temperature"] = args.temperature
    
    key_manager = KeyManager(API_KEYS)
    results = []
    already_done = set()
    
    try:
        with open(args.output_file, "r") as f:
            for line in f:
                already_done.add(json.loads(line.strip())["key"])
    except FileNotFoundError:
        pass

    tasks = []
    for index, (key, prompt) in enumerate(prompts.items()):
        if key in already_done:
            print(f"Skipping {key}")
            continue
        tasks.append(
            asyncio.create_task(
                process_job(key_manager, args, prompt, key, index, results, system_instruction, generation_config)
            )
        )

    await asyncio.gather(*tasks)

    with open(f"{args.input_file}.combined_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Processing complete. Results saved.")

if __name__ == "__main__":
    asyncio.run(main())