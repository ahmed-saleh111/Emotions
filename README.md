# School Video Analysis

A professional video analysis tool for emotion detection in school environments, built with Streamlit and OpenCV. This application enables users to upload classroom videos and automatically process them to detect emotions frame-by-frame using a finetuned YOLOv11 model trained on an emotion dataset.


## Features

- **Video Upload**: Supports MP4 and AVI formats for easy classroom video uploads.
- **Emotion Detection**: Processes each frame to detect emotions using a pre-trained model.
- **YOLOv11 Integration**: Utilizes pre-trained YOLOv11 models from Ultralytics for robust face and emotion detection.
- **Real-Time Visualization**: Displays processed frames in real-time during analysis.
- **Processed Video Output**: Saves and allows playback of the processed video with detected emotions overlayed.

## Getting Started

### Prerequisites
- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)
- [Ultralytics YOLO](https://docs.ultralytics.com/) (for YOLOv11 models)
- Other dependencies as required by your models (see below)



### Installation
It is recommended to use a dedicated conda environment for this project. Below are the steps to set up the environment with Python 3.12:

1. Create and activate a new conda environment:
   ```bash
   conda create -n emotion-env python=3.12 -y
   conda activate emotion-env
   ```

2. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd emotion
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your trained models in the `models/` directory.

### Usage
1. Start the Streamlit app:
   ```bash
   streamlit run streamlit.py
   ```
2. Upload a classroom video (MP4 or AVI).
3. Click **Start processing** to analyze the video.
4. View real-time results and download the processed video.

## Project Structure

```
.
├── emotin.py                # Emotion detection logic
├── streamlit.py             # Streamlit web application
├── models/                  # Pre-trained model files (including YOLOv11 from Ultralytics)
├── videos/                  # Sample videos
└── ...
```
![Emotion Detection](attachment://https://github.com/ahmed-saleh111/Emotions/blob/main/emotion.gif))

## Model Files
- Place your PyTorch or ONNX models in the `models/` directory.
- Example model files:
  - `best.pt`
  - `yolo11s-emotion.pt` (YOLOv11 emotion model from Ultralytics)
  - `yolov11n-face.pt` (YOLOv11 face detection model from Ultralytics)

## Using Pretrained YOLOv11 from Ultralytics
- This project leverages pretrained YOLOv11 models from [Ultralytics](https://github.com/ultralytics/ultralytics) for face and emotion detection.
- You can download official YOLOv11 models or train your own using the Ultralytics framework.
- To use a pretrained model, place the `.pt` file in the `models/` directory and ensure your code loads it with the Ultralytics API:
  ```python
  from ultralytics import YOLO
  model = YOLO('models/yolo11s-emotion.pt')
  results = model(frame)
  ```
- For more details, see the [Ultralytics YOLO documentation](https://docs.ultralytics.com/).


## License
This project is intended for educational and research purposes. Please review and comply with your institution's data privacy and ethical guidelines when using classroom videos.

