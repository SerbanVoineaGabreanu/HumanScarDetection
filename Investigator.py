#Human Scar Detection (Industry Application Project)
#COSC 5437 Neural Networking
#
#Step 2
#
#Fangze Zhou & Serban Voinea Gabreanu
#This investigator script is responsible for managing the LLM and GUI that allows a 
#user to find out the origin of their scar, or even diagnose the type of scar.
#(Although step 1 should be more accurate at diagnosing the type of scar, so
#it is recommended the users start with step 1.)

import os
import base64
import datetime
import json
import threading
from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from pathlib import Path
from io import BytesIO
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
#Note that the program was mainly tested with gemma 3 27B Q4, medgemma 27B Q4 was also partially tested but with the existing prompt it's going to 
#produce responses that are, a little bit rude. The program may technically be used with an .gguf formated model but it has only been tested to work properly
#with the gemma-3-27B-it-QAT-Q4_0.gguf model, the system prompt may require tuning or a complete re-write for a different model (e.g. for medgemma).
VISION_MODEL_PATH  = BASE_DIR / "LLMs" / "gemma-3-27B-it-QAT-Q4_0.gguf"
VISION_MMPROJ_PATH = BASE_DIR / "LLMs" / "mmproj-model-f16.gguf"
#VISION_MODEL_PATH  = BASE_DIR / "LLMs" / "medgemma-27b-it-Q4_0.gguf"
#VISION_MMPROJ_PATH = BASE_DIR / "LLMs" / "mmproj-F32.gguf"
CONVERSATION_SAVE_PATH = BASE_DIR / "Conversations" / "Scarface"

#Options for the LLM, GPU layers should be set to -1 so the GPU is fully loaded to 
#run the model quicker, unless the system has a very powerful CPU with a lot of RAM an a weak/low VRAM GPU.
#Context window will vary depending on amount of RAM/VRAM available, currently set to 16k with the 30B Q4 Gemma model uses 
#around 24GB of RAM. 
N_GPU_LAYERS = -1
N_CTX = 16384
LLM_REPLY_TOKEN_LIMIT = 1024

### System Prompt ###
#This is very important, and it tells the LLM how to behave and gives it some rules it needs to follow.
#In this case the LLM is given a name "Scarface", and an objective, and some preliminary information about scars,
#although the LLM should have scar information in its parameters, this give it a minimum amount of knowledege (very tiny
#models may have very little medical knowledge for example)
#Tiny changes to the system prompt can completely change the model's behaviour, and if the model is changed the system
#prompt might also need to be altered, this prompt was optimized for Gemma 3 27B Quantized at Q4.
SCARFACE_SYSTEM_PROMPT = """
You are Scarface, a specialized AI known as a "Scar Investigator." Your one and only purpose is to help users understand the potential origin of their scars. You must strictly adhere to this persona. Your name is Scarface, and you will not change it.

**Your Goal:**
To analyze the information provided by the user (scar diagnosis, images, and descriptions) and determine the most likely cause of their scar.

**Your Behavior:**
1.  **Analyze First:** If you receive a diagnosis or an image at the start, begin your investigation immediately. Else introduce yourself and explain to the user your task.
2.  **Be Inquisitive:** If the cause is not obvious, ask targeted questions to gather more information. For example, ask about the circumstances of the injury, the healing process, or any known medical conditions.
3.  **Be Helpful and Empathetic:** The user may be distressed. Be supportive and guide them through the investigation process.
4.  **Stay Focused:** You are forbidden from deviating from your role as a Scar Investigator. Do not discuss other topics, write poetry, or engage in unrelated conversation.
5.  **Use Your Knowledge:** You have been provided with supplemental knowledge about common scar types. Use this to inform your questions and conclusions.
6.  **Be Concise** You must explain in a simple and short manner, do not ramble.

- **Atrophic:** Indented; tissue loss (acne, chickenpox, surgery). Subtypes: Icepick, Boxcar, Rolling.
- **Hypertrophic:** Raised; excess collagen; within wound (piercings, cuts, burns, surgery).
- **Keloid:** Severely raised; extends beyond wound (minor injuries, genetics).
- **Contracture:** Skin tightening; movement impairment (burns, large wounds, joints).
- **Stretch Marks (Striae):** Skin stretching/shrinking; collagen/elastin rupture (pregnancy, weight changes, growth spurts).
- **Acne:** Various types from deep blemishes (icepick, boxcar, rolling, sometimes hypertrophic/keloid).
- **Surgical:** From incisions; varies (any surgery).
- **Burn:** From burns; type depends on depth (contracture, hypertrophic, discolored).
- **C-Section:** Specific surgical scar (cesarean delivery).
- **Traumatic:** From accidental injuries (cuts, scrapes, lacerations).
- **Vaccination:** Small, round from injections.
- **Self-Harm:** From intentional injury (cutting, burning); often linear, parallel.
- **Piercing:** At piercing sites (small to hypertrophic/keloid).
- **Hypopigmented:** Lighter; melanin loss (burns, inflammation).
- **Hyperpigmented:** Darker; excess melanin (darker skin tones, inflammation, sun).
- **Maturational:** Healing/changing phase; initially red/raised, then fades.

You are now ready to begin the investigation.
"""

### Flask Application ###
#Allows the script to run on a web browser like a normal website.
app = Flask(__name__)

llm = None
model_loading_lock = threading.Lock()
initial_data = None

#The LLM is loaded. A LVM capable model (Gemma 3 in this case) is used so the GGUF and MMProj file are both loaded.
def get_llm():
    global llm
    with model_loading_lock:
        if llm is None:
            print("Loading Scarface LLM for the first time...")
            try:
                if not os.path.exists(VISION_MODEL_PATH):
                    raise FileNotFoundError(f"Vision Model GGUF not found: {VISION_MODEL_PATH}")
                if not os.path.exists(VISION_MMPROJ_PATH):
                    raise FileNotFoundError(f"MMProj file not found: {VISION_MMPROJ_PATH}")

                chat_handler = Llava15ChatHandler(clip_model_path=str(VISION_MMPROJ_PATH), verbose=True)
                llm = Llama(
                    model_path=str(VISION_MODEL_PATH),
                    chat_handler=chat_handler,
                    n_ctx=N_CTX,
                    n_gpu_layers=N_GPU_LAYERS,
                    verbose=True
                )
                print("Scarface LLM loaded successfully.")
            except Exception as e:
                print(f"FATAL: The LLM could not be loaded. Error: {e}")
                llm = None
    return llm

@app.before_request
def ensure_model_is_loaded():
    if request.endpoint not in ['exit_app', 'static']:
        get_llm()

### Web Page Code ###

@app.route('/')
def index():
    return render_template('index.html')

#Recieves the initial diagnostics data from the Manager.py script. 
@app.route('/initiate_diagnosis', methods=['POST'])
def initiate_diagnosis():
    global initial_data
    data = request.json
    diagnosis = data.get('diagnosis_text')
    image_b64 = data.get('image_base64')
    
    if not diagnosis and not image_b64:
        return jsonify({"status": "error", "message": "No diagnosis text or image provided."}), 400
        
    print(f"Received initial diagnosis data: Text='{diagnosis}', Image provided='{image_b64 is not None}'")
    # Store the data in the global variable to be used by the first /chat call
    initial_data = {"diagnosis": diagnosis, "image": image_b64}
    
    return jsonify({"status": "success", "message": "Diagnosis data received and stored for the session."}), 200
#Handles the main chat logic, robustly supporting all conversation scenarios:
#Manager Start: Merges system prompt for the LLM but shows a clean message to the user.
#Direct Start: Uses a meta-prompt to make the LLM introduce itself.
#Ongoing Chat: Uses the full chat history to maintain context.
#Image Handling: Converts any uploaded images to a compatible JPEG format.
@app.route('/chat', methods=['POST'])
def chat():
    global initial_data
    scarface_llm = get_llm()
    if not scarface_llm:
        return jsonify({"error": "LLM is not available. Check server logs."}), 503

    data = request.json
    client_history = data.get('history', [])
    messages_to_send = []

    gen_params = {
        "temperature": 0.7,
        "repeat_penalty": 1.1,
        "stop": ["USER:", "User:", "\nUSER:", "\nUser:", "ASSISTANT:", "Assistant:"]
    }

    final_history_for_client = list(client_history)

    #Checks to see if this is the first turn of the conversation
    if not client_history:
        #SCENARIO A: The conversation is started by the Manager script
        if initial_data:
            print("Manager flow: Creating separate messages for LLM and Client display.")
            text_for_user = "Let's begin the investigation. My scar has been classified as: '{}'. Please analyze this information and the attached image to start.".format(initial_data.get('diagnosis', 'Unknown'))
            text_for_llm = f"{SCARFACE_SYSTEM_PROMPT}\n\n---\n\n{text_for_user}"

            content_for_client = [{"type": "text", "text": text_for_user}]
            content_for_llm = [{"type": "text", "text": text_for_llm}]
            
            if initial_data.get("image"):
                image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{initial_data['image']}"}}
                content_for_client.append(image_content)
                content_for_llm.append(image_content)

            #Appends the correct version of the messages to each list
            final_history_for_client.append({"role": "user", "content": content_for_client})
            messages_to_send.append({"role": "user", "content": content_for_llm})
            
            initial_data = None
        
        #SCENARIO B: The user starts the chat directly in the Investigator UI
        else:
            print("First contact from UI. Crafting a meta-prompt for the initial greeting.")
            greeting_request = "Introduce yourself as Scarface, the Scar Investigator, and ask me to tell you about my scar to begin the investigation."
            #The LLM gets the full system prompt plus the instruction to introduce itself
            prompt_for_llm = f"{SCARFACE_SYSTEM_PROMPT}\n\n---\n\n{greeting_request}"
            messages_to_send.append({"role": "user", "content": prompt_for_llm})
            gen_params["temperature"] = 0.5

    #SCENARIO C: This is a next turn in an on going conversation
    else:
        user_input = data.get('message', '')
        image_b64 = data.get('image')

        if image_b64:
            try:
                image_bytes = base64.b64decode(image_b64)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                output_buffer = BytesIO()
                image.save(output_buffer, format="JPEG")
                image_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
                print("Successfully converted chat image to JPEG.")
            except Exception as e:
                print(f"Error converting chat image: {e}. Image will be ignored.")
                image_b64 = None

        user_content = [{"type": "text", "text": user_input}]
        if image_b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        
        new_user_message = {"role": "user", "content": user_content}
        final_history_for_client.append(new_user_message)
        
        #The LLM gets the entire conversation history in order to maintain the context
        messages_to_send.extend(final_history_for_client)

    try:
        response = scarface_llm.create_chat_completion(
            messages=messages_to_send,
            temperature=gen_params["temperature"],
            repeat_penalty=gen_params["repeat_penalty"],
            stop=gen_params["stop"],
            max_tokens=LLM_REPLY_TOKEN_LIMIT
        )
        assistant_response = response['choices'][0]['message']['content'].strip()

        final_history_for_client.append({"role": "assistant", "content": assistant_response})
        
        return jsonify({"reply": assistant_response, "history": final_history_for_client})

    except Exception as e:
        print(f"Error during LLM chat completion: {e}")
        return jsonify({"error": "Failed to get a response from the LLM."}), 500

@app.route('/next_step', methods=['POST'])
def next_step():
    scarface_llm = get_llm()
    if not scarface_llm:
        return jsonify({"error": "LLM is not available. Check server logs."}), 503

    data = request.json
    history = data.get('history', [])

    summary_prompt = "Based on our entire conversation, please write a final summary titled 'Final Story'. This summary should explain your final conclusion on how the user likely got their scar. Be concise and definitive in your conclusion. Also create a cool idea for a tattoo to cover up their scar."

    messages_for_summary = [
        {"role": "system", "content": SCARFACE_SYSTEM_PROMPT},
    ] + history + [
        {"role": "user", "content": summary_prompt}
    ]

    try:
        response = scarface_llm.create_chat_completion(
            messages=messages_for_summary,
            temperature=0.5,
            repeat_penalty=1.1,
            stop=["USER:", "User:", "\nUSER:", "\nUser:"],
            max_tokens=LLM_REPLY_TOKEN_LIMIT
        )
        final_story = response['choices'][0]['message']['content'].strip()

        os.makedirs(CONVERSATION_SAVE_PATH, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(CONVERSATION_SAVE_PATH, f"Investigation_{timestamp}.txt")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Scarface Investigation Log - {timestamp}\n")
            f.write("="*40 + "\n\n")
            for message in history:
                role = "User" if message['role'] == 'user' else "Scarface"
                content = ""
                if isinstance(message['content'], list):
                    for item in message['content']:
                        if item['type'] == 'text':
                            content += item['text']
                else:
                    content = message['content']
                f.write(f"--- {role} ---\n{content}\n\n")

            f.write("="*40 + "\n")
            f.write(f"{final_story}\n")

        print(f"Conversation saved to {filename}")
        return jsonify({"status": "success", "message": f"Conversation saved. {final_story}", "final_story": final_story})

    except Exception as e:
        print(f"Error during next_step processing: {e}")
        return jsonify({"error": "Failed to generate final story or save file."}), 500


### Step 3 Starter Code ### (This should eventually lead to Step 3.)
@app.route('/start_step3', methods=['POST'])
def start_step3():
    data = request.json
    final_diagnosis = data.get('final_diagnosis')
    images_b64 = data.get('images', []) 

    if not final_diagnosis:
        return jsonify({"status": "error", "message": "Final diagnosis text is required."}), 400

    print("\n### RECEIVED DATA FOR STEP 3 (Tattoo Suggester) ###")
    print(f"Final Diagnosis/Story: {final_diagnosis}")
    print(f"Number of images received: {len(images_b64)}")
    print("### (Note that is a placeholder, no actual script was called) ###\n")

    return jsonify({"status": "success", "message": "Data for Step 3 received successfully."})


@app.route('/exit', methods=['POST'])
def exit_app():
    print("Exit request received. Shutting down server.")
    os._exit(0)

if __name__ == '__main__':
    print("### Scarface Investigator Server Starting ###")
    print(f"Model Path: {VISION_MODEL_PATH}")
    print(f"Conversation Save Directory: {CONVERSATION_SAVE_PATH}")
    print("Starting Flask server...")
    print("The LLM will be loaded into memory upon the first web request.")
    print("Access the application at http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)