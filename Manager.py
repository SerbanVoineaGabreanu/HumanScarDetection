#Manager.py
#Human Scar Detection (Industry Application Project)
#COSC 5437 Neural Networking
#Fangze Zhou & Serban Voinea Gabreanu
#
#Step 1B
#
#Description: This script acts as the initial entry for the user. 
#It opens a website where the user either can upload their picture or skip
#straight to talking with an LLM.

import os
import torch
import timm
import json
import base64
import subprocess
import requests
import time
import sys 
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
from pathlib import Path
from io import BytesIO

#Note that the Models.py script has to be in the same folder as this script.
from Models import create_modern_cnn, _VARIANTS

BASE_DIR = Path(__file__).resolve().parent
DEPLOY_DIR = BASE_DIR / 'DeployModel'
PROCESSED_DATASET_DIR = BASE_DIR / 'ProcessedDataset'
DEFAULT_VARIANT = "tiny"
INVESTIGATOR_SCRIPT_PATH = BASE_DIR / 'Investigator.py'
INVESTIGATOR_URL = "http://127.0.0.1:5001"

app = Flask(__name__, template_folder=str(BASE_DIR))

loaded_model = None
class_names = None
model_transform = None
investigator_process = None

### Model Loading Methods ###
def find_latest_model():
    if not os.path.exists(DEPLOY_DIR):
        return None
    
    models = [f for f in os.listdir(DEPLOY_DIR) if f.endswith('.pth') or f.endswith('.pth.tar')]
    if not models:
        return None

    latest_model = max(models, key=lambda m: os.path.getmtime(os.path.join(DEPLOY_DIR, m)))
    return os.path.join(DEPLOY_DIR, latest_model)

#Checks if GPU acceleration can be used.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

#Responsible for loading the classification model.
def load_classification_model():
    global loaded_model, class_names, model_transform

    model_path = find_latest_model()
    if not model_path:
        print("Error: No deployed model has been found in the 'DeployModel' folder!")
        return False

    class_mapping_path = os.path.join(PROCESSED_DATASET_DIR, 'class_mapping.json')
    if not os.path.exists(class_mapping_path):
        print(f"Error: 'class_mapping.json' not found in '{PROCESSED_DATASET_DIR}'.")
        print("Please make sure that the preprocessing step in DL_Training_Scars.py has been run.")
        return False
        
    with open(class_mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    
    device = get_device()
    print(f"Loading model '{os.path.basename(model_path)}' onto device '{device}'...")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        variant = checkpoint.get('variant', DEFAULT_VARIANT)
        use_tl = checkpoint.get('use_transfer_learning', False)

        if use_tl:
            model = timm.create_model(f'convnext_{variant}', pretrained=False, num_classes=num_classes)
        else:
            model = create_modern_cnn(num_classes=num_classes, variant=variant)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        loaded_model = model
        model_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("Model loaded successfully.")
        return True
    except Exception as e:
        print(f"ERROR: Model could not be loaded! Error: {e}")
        loaded_model = None
        class_names = None
        return False

### Flask Methods ###

#Renders the main page.
@app.route('/')
def manager_home():
    return render_template('manager.html')

#Responsible for recieving the image and the classifying it, and then providing a result to the user.
@app.route('/classify', methods=['POST'])
def classify_image():
    if not loaded_model:
        if not load_classification_model():
            return jsonify({"Error": "The model could not be loaded!"}), 500

    if 'image' not in request.files:
        return jsonify({"Error": "No image file was given!"}), 400

    file = request.files['image']
    try:
        image = Image.open(file.stream).convert("RGB")
        image_tensor = model_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(get_device())

        with torch.no_grad():
            outputs = loaded_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item() * 100

        print(f"Classification result: {predicted_class} with {confidence_score:.2f}% confidence.")
        return jsonify({"result": predicted_class, "confidence": f"{confidence_score:.2f}%"})
    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({"error": "Failed to process the image."}), 500

#Investigator.py script is launched as a subprocess, which helps to automate the process 
#for the user, so they don't have to send the results to the LLM themselves.
@app.route('/launch_investigator', methods=['POST'])
def launch_investigator():
    global investigator_process
    if investigator_process and investigator_process.poll() is None:
        print("Investigator script is already running.")
        return jsonify({"status": "already_running", "url": INVESTIGATOR_URL})

    try:
        print(f"Launching investigator script with interpreter: {sys.executable}")
        #Uses the sys.executable to ensure the correct python environment is used (this specific script was tested only on a MacOS system).
        investigator_process = subprocess.Popen([sys.executable, str(INVESTIGATOR_SCRIPT_PATH)])
        
        #Depending on the LLM and the hardware, it can take a while to load, this may need to be altered if it takes too long.
        print("Waiting for investigator server to initialize...")
        time.sleep(15)
        
        print("Investigator script launched.")
        return jsonify({"status": "success", "url": INVESTIGATOR_URL})
    except Exception as e:
        print(f"Failed to launch investigator script: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

#Launches the Investigator.py script and sends it the initial classification data. It also converts any image to JPEG, since some 
#images like WebP are NOT compatible with the LLM.
@app.route('/next_step', methods=['POST'])
def next_step_to_investigator():
    data = request.json
    diagnosis_text = data.get('diagnosis')
    image_b64 = data.get('image_base64')

    if not diagnosis_text:
        return jsonify({"status": "error", "message": "Missing diagnosis data."}), 400

    #Images conversion. 
    if image_b64:
        try:
            image_bytes = base64.b64decode(image_b64)
            image_stream = BytesIO(image_bytes)
            image = Image.open(image_stream)

            #Converts the image into RGB format.
            image = image.convert("RGB")

            output_buffer = BytesIO()
            image.save(output_buffer, format="JPEG")
            jpeg_bytes = output_buffer.getvalue()

            #Reencode the clean JPEG bytes into a new Base64 string for the LLM.
            image_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
            print("Successfully converted uploaded image to JPEG.")

        except (base64.binascii.Error, UnidentifiedImageError) as e:
            print(f"Error processing image: {e}. It might be corrupted or not a valid image format.")
            return jsonify({"status": "error", "message": "The uploaded file is not a valid or supported image."}), 400
        except Exception as e:
            print(f"An unexpected error occurred during image conversion: {e}")
            return jsonify({"status": "error", "message": "An internal error occurred while processing the image."}), 500

    launch_response = launch_investigator()
    #Checks to see if launching the investigator was successful.
    launch_data = launch_response.get_json()
    if launch_response.status_code != 200 and launch_data.get('status') not in ['already_running']:
         return jsonify({"status": "error", "message": f"Could not start the investigator service: {launch_data.get('message', 'Unknown error')}"}), 500

    if launch_data.get('status') not in ['success', 'already_running']:
        return jsonify({"status": "error", "message": "Could not start the investigator service."}), 500

    initiate_url = f"{INVESTIGATOR_URL}/initiate_diagnosis"
    payload = {"diagnosis_text": diagnosis_text, "image_base64": image_b64}

    try:
        print("Sending initial data to investigator...")
        response = requests.post(initiate_url, json=payload, timeout=20)

        if response.status_code == 200:
            print("Successfully sent data to investigator.")
            return jsonify({"status": "success", "url": INVESTIGATOR_URL})
        else:
            error_message = response.json().get('message', f"Investigator service responded with status {response.status_code}")
            return jsonify({"status": "error", "message": error_message}), 500
    except requests.exceptions.Timeout:
        return jsonify({"status": "error", "message": "Timed out waiting for the investigator service. The LLM may be taking too long to load."}), 500
    except requests.exceptions.ConnectionError:
        return jsonify({"status": "error", "message": "Connection refused by the investigator service. Please check if it is running correctly."}), 500
    except Exception as e:
        print(f"An error occurred while sending data to the investigator script: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

#Process to shut down the manager and the investigator script if it also currently running.
@app.route('/exit_manager', methods=['POST'])
def exit_manager():
    global investigator_process
    print("Exit request received. Shutting down all services.")
    
    if investigator_process and investigator_process.poll() is None:
        print("Terminating investigator process...")
        investigator_process.terminate()
        investigator_process.wait()
        print("Investigator process terminated.")
        
    os._exit(0)

if __name__ == '__main__':
    print("--- Scar Classification Manager Starting ---")
    print(f"Looking for models in: {DEPLOY_DIR}")
    print(f"Looking for class mappings in: {PROCESSED_DATASET_DIR}")
    print("Starting Flask server for the manager...")
    print("Access the application at http://127.0.0.1:5002")
    load_classification_model()
    app.run(host='0.0.0.0', port=5002, debug=False)