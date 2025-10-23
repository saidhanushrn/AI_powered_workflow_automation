from flask import Flask, request, jsonify
from pyngrok import ngrok
import requests
import os

app = Flask(__name__)
os.environ['HF_API_TOKEN'] = 'API_TOKEN'
API_TOKEN = os.getenv('HF_API_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/distilbert/distilbert-base-uncased-finetuned-sst-2-english"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def classify_text(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    try:
        return response.json()
    except Exception as e:
        print("Error parsing JSON:", e)
        return None

def map_to_triage(model_output):
    if not model_output or not model_output[0]:
        return 'Needs Review', 'Medium'

    top_prediction = model_output[0][0]
    label = top_prediction['label']
    score = top_prediction['score']

    if label == 'NEGATIVE':
        if score > 0.85:
            return 'Technical Issue', 'High'
        elif score > 0.6:
            return 'Technical Issue', 'Medium'
        else:
            return 'Needs Review', 'Medium'

    elif label == 'POSITIVE':
        if score > 0.85:
            return 'General Inquiry', 'Low'
        elif score > 0.6:
            return 'General Inquiry', 'Medium'
        else:
            return 'Needs Review', 'Medium'


@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    text = data.get('text', '')
    model_output = classify_text(text)
    category, priority = map_to_triage(model_output)
    return jsonify({"category": category, "priority": priority})

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print("Ngrok URL:", public_url)
    app.run(port=5000)
