# CCTV Anomaly Detection Flask Application

This Flask application performs anomaly detection on video streams (webcam or uploaded videos) using a pre-trained ResNetLSTM model. The model can detect various anomalies including Arrest, Assault, Explosion, Road Accidents, Stealing, and Vandalism.

## Features

- Real-time webcam processing with anomaly detection
- Upload and process pre-recorded videos
- Visual display of detection results with confidence scores
- Console logging of detection events
- Responsive web interface

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- Flask
- PIL (Python Imaging Library)
- NumPy

## Installation

1. Clone this repository:
```
https://github.com/Poudel0/Anomaly-Inference
```

2. Install required packages:
```
pip install torch torchvision opencv-python flask pillow numpy
```

3. Place your trained model checkpoint in the project root directory:
```
!todo```

## Usage

1. Start the Flask application:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Choose your input source:
   - **Webcam**: Uses your default webcam for real-time detection
   - **Upload Video**: Upload a pre-recorded video file (.mp4, .avi, .mov, .mkv)

## Model Details

The application uses a ResNetLSTM model trained for multi-label anomaly detection with the following parameters:

- **Input Dimensions**: 64x64 RGB images
- **Frame Rate**: 3 FPS (processes every 10th frame from a standard 30 FPS video)
- **Sequence Length**: 32 frames are processed together for each prediction
- **Detection Threshold**: 0.4 (probability threshold for positive detection)
- **Classes**: NormalVideos, Arrest, Assault, Explosion, RoadAccidents, Stealing, Vandalism

## Directory Structure

```
├── app.py                     # Main Flask application
├── templates/                 # HTML templates
│   └── index.html             # Web interface
├── uploads/                   # Directory for uploaded videos
├── model_checkpoint_epoch_7.pth  # Trained model weights
└── README.md                  # This file
```

## Notes

1. The model processes video at 3 FPS regardless of the original video's frame rate
2. Each prediction is based on a sequence of 32 frames
3. The model resizes all frames to 64x64 pixels before processing
4. The application's performance depends on your hardware (CPU/GPU)
5. For real-time processing, a GPU is recommended but not required

## Limitations

1. The model's accuracy depends on the quality of the training data
2. Low-light conditions may reduce detection accuracy
3. Processing large video files may be memory-intensive
4. The web interface may experience lag with slow connections

## Future Improvements

1. Add support for multiple webcams
2. Implement recording capability for detected anomalies
3. Add email/SMS notifications for detected anomalies
4. Improve UI with more detailed analytics 
5. Add batch processing for multiple video files
