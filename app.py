from flask import Flask, request, render_template
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")


t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")


def analyze_image(image_path):
    image = Image.open(image_path)
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = blip_model.generate(**inputs)
    description = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return description


def generate_test_instructions(description):
    prompt = f"Based on the following screenshot description: '{description}', generate a detailed test case with: " \
             f"1. Description: Brief summary of the test case. " \
             f"2. Pre-conditions: Requirements that need to be set up before testing. " \
             f"3. Testing Steps: Step-by-step instructions for testing. " \
             f"4. Expected Result: What should happen if the feature works correctly. " \
             f"5. Test Results: Possible outcomes based on the results."

    inputs = t5_tokenizer(prompt, return_tensors="pt")
    outputs = t5_model.generate(inputs.input_ids, max_length=300)
    test_case = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return test_case

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate-instructions', methods=['POST'])
def generate_instructions():
    context = request.form.get('context', '')
    images = request.files.getlist('images')

    if not images:
        return render_template('index.html', error='No images provided. Please upload at least one screenshot.')

    descriptions = []
    for image in images:
        try:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            description = analyze_image(image_path)
            descriptions.append(description)
        except Exception as e:
            print(f"Error processing image {image.filename}: {e}")
            return render_template('index.html', error=f"Error processing image {image.filename}. Please try again.")


    instructions = [generate_test_instructions(description) for description in descriptions]

    return render_template('index.html', instructions=instructions)

if __name__ == '__main__':
    app.run(debug=True)

