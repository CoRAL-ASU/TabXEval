import os
from openai import OpenAI
from dotenv import load_dotenv
import sys 
from argparse import ArgumentParser
sys.path.append('/home/turning/Jainit/TANQ')
from src.utils.interative_alignment import merge_tables_fuzzy
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
import json

def ask_chatgpt(text: str):
    # Check if a prompt path is provided and read prompt text
    _ = load_dotenv()
    api_key =os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key = api_key)


    # Model configuration - replace 'gpt-4' with the specific model if needed
    model_name = "gpt-4o"  # or "gpt-3.5-turbo" if you want a different model
    # Send the prompt to the model
    response = client.chat.completions.create(
    model=model_name,
    messages=[
        {'role': "user" , "content": text}
    ],
    temperature=0.1, 
    top_p=0.2
    )
        
    # Return the response content
    return response.choices[0].message.content


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, help='Input text to send to the model')
    parser.add_argument('--output', type=str, help='Output file to save the results')
    
    args = parser.parse_args()
    prompts = json.load(open(args.input))
    
    results = []
    for uid, prompt in prompts.items():
        try: 
            response = ask_chatgpt(prompt)
            print(f"RESPONSE: {response}")
            results.append({'uid': uid, 'response': response, 'prompt': prompt})
        except Exception as e:
            print(f"Error: {e}")
            results.append({'uid': uid, 'response': "Error", 'prompt': prompt})
    
    # Save the results to a file
    json.dump(results, open(args.output, 'w'))