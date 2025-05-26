import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from collections import deque
import threading
import time
import queue
import requests
from dotenv import load_dotenv

import os
IFTTT_WEBHOOK_URL = os.getenv("IFTTT_WEBHOOK_URL")

last_alert_time = 0
alert_interval = 100  # seconds


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model definition
class ResNetLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=256, lstm_num_layers=2, dropout=0.3):
        super(ResNetLSTM, self).__init__()
        resnet = models.resnet50(weights=None)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.lstm = nn.LSTM(2048, lstm_hidden_size, lstm_num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        features = self.resnet(c_in)
        features = features.view(batch_size, timesteps, -1)
        lstm_out, _ = self.lstm(features)
        lstm_out = self.dropout(lstm_out[:, -1, :])
        logits = self.fc(lstm_out)
        output = self.sigmoid(logits)
        return output

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

classes = ['NormalVideos', 'Arrest', 'Assault', 'Explosion', 'RoadAccidents', 'Stealing', 'Vandalism']

sequence_length = 32
threshold = 0.4
frame_queue = deque(maxlen=sequence_length)
result_queue = queue.Queue(maxsize=10)
processing = False
output_frame = None
current_results = {}
current_video_filename = ""


def send_alert(alert_class):
    global last_alert_time

    if not IFTTT_WEBHOOK_URL:
        print("No IFTTT webhook URL configured.")
        return

    now = time.time()
    if now - last_alert_time < alert_interval:
        print(f"Alert skipped: {alert_class} (within 15s interval)")
        return

    try:
        message = alert_class
        response = requests.post(IFTTT_WEBHOOK_URL, json={"Anomaly": message})
        if response.status_code == 200:
            print(f"Alert sent: {message}")
            last_alert_time = now
        else:
            print(f"Alert failed: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Initialize model
def initialize_model():
    global model, transform
    model = ResNetLSTM(num_classes=len(classes)).to(device)
    try:
        checkpoint = torch.load('output/checkpoint_epoch_8.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Process frames
def process_frames():
    global frame_queue, processing, output_frame, current_results, current_video_filename
    frame_counter = 0

    while processing:
        if len(frame_queue) == sequence_length:
            if frame_counter % sequence_length == 0:
                sequence = []
                for frame in frame_queue:
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    t_image = transform(pil_image)
                    sequence.append(t_image)

                input_tensor = torch.stack(sequence).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)

                probs = outputs[0].cpu().numpy()
                results = {}

                for i, cls in enumerate(classes):
                    prob = float(probs[i])
                    results[cls] = prob

                top_classes = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
                current_results = {
                    cls: {'probability': prob, 'detected': prob >= threshold}
                    for cls, prob in top_classes
                }

                print("\nDetection Results:")
                for cls, result in current_results.items():
                    print(f"- {cls}: {result['probability']:.2f} confidence")
                for cls, result in current_results.items():
                      if cls != 'NormalVideos' and result['detected']:
                       send_alert(cls)
                       break  # Alert once per sequence

                try:
                    if not result_queue.full():
                        result_queue.put(current_results.copy(), block=False)
                except queue.Full:
                    pass

            frame_counter += 1
        time.sleep(0.01)

# Helper

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

# Process uploaded video

def process_video_file(video_path):
    global frame_queue, processing, output_frame, current_results

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps / 3)

    frame_count = 0
    processing = True

    processing_thread = threading.Thread(target=process_frames)
    processing_thread.daemon = True
    processing_thread.start()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_skip == 0:
                resized_frame = cv2.resize(frame, (64, 64))
                frame_queue.append(resized_frame)
                output_frame = frame.copy()

                y_pos = 30
                for cls, result in current_results.items():
                    color = (0, 0, 255) if cls != 'NormalVideos' else (0, 255, 0)
                    text = f"{cls}: {result['probability']:.2f}"
                    cv2.putText(output_frame, text, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    y_pos += 30

                ret, buffer = cv2.imencode('.jpg', output_frame)
                if not ret:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            frame_count += 1
            time.sleep(1.0/30.0)

    finally:
        processing = False
        cap.release()

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('path', '')
    if video_path and os.path.exists(video_path):
        return Response(process_video_file(video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Video file not found", 404

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_video_filename
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_video_filename = filename

        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/results')
def get_results():
    try:
        if not result_queue.empty():
            results = result_queue.get(block=False)
            return jsonify(results)
        else:
            return jsonify(current_results)
    except:
        return jsonify({})

if __name__ == '__main__':
    load_dotenv()
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000)