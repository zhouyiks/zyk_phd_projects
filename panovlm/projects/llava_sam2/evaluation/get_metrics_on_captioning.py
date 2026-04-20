import os
import json
import numpy as np
import base64
from openai import OpenAI


def main():
    prediction_root = "./eval_results/chatgpt4o/captioning"
    save_root = "./eval_results/chatgpt4o/captioning_metrics"

    MODEL='chatgpt-4o-latest'
    client = OpenAI(api_key='')
    for json_file in os.listdir(prediction_root):
        json_path = os.path.join(prediction_root, json_file)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        case_id = json_data['case_id']
        positive_phrases = json_data['positive_negative']['positive']
        prediction = json_data['prediction']

        save_path = os.path.join(save_root, f"{case_id}.json")
        if os.path.exists(save_path):
            continue
        
        caption = "{" + prediction + "}"
        question = f"I used a model to generate a caption for an entity in an image: {caption}\n\n"\
            f"I have a set of phrases: {positive_phrases} that describe some attributes of this entity, "\
            "which can be considered as ground truth. Please help me evaluate the quality of "\
            "this generated caption. For each phrase, determine whether it is mentioned in the "\
            "generated caption, returning 1 if mentioned and 0 if not. Finally, return a list "\
            "of 0s and 1s with the same length as the phrases list."
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": [{"type": "text", "text": question},]}
            ]
        )
        output_text = response.choices[0].message.content
        print(output_text)
        exit(0)




if __name__ == "__main__":
    main()