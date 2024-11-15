# app.py
from flask import Flask, request, jsonify
import torch
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# Load models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
chat_model = GPT2LMHeadModel.from_pretrained("gpt2")
chat_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get image file from request
    image_file = request.files['image']
    
    # Preprocess and predict with CLIP
    inputs = clip_processor(images=image_file, return_tensors="pt")
    clip_outputs = clip_model(**inputs)
    
    # Convert CLIP output to text description
    clip_text_output = "Interpretation of image based on model"
    
    # Generate chatbot response
    chat_input = chat_tokenizer.encode(f"User: {clip_text_output}\nAI:", return_tensors='pt')
    chat_output = chat_model.generate(chat_input, max_length=100, do_sample=True)
    response_text = chat_tokenizer.decode(chat_output[0], skip_special_tokens=True)
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run()
