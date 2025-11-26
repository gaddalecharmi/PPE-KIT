# PPE Detection System - Desktop Application

A complete Personal Protective Equipment (PPE) detection system using YOLOv8, OpenCV, and Streamlit. Detects helmets, gloves, safety shoes, safety vests, and safety glasses in images, videos, and live webcam feeds.

## Features

✅ **Three Input Modes:**
- 📷 Upload and analyze images
- 🎥 Upload and process videos
- 📹 Real-time webcam detection

✅ **Detects 5 Types of PPE:**
- 🪖 Helmets
- 🧤 Gloves
- 👟 Safety Shoes
- 🦺 Safety Vests
- 🥽 Safety Glasses

✅ **Visual Features:**
- Color-coded bounding boxes
- Confidence scores
- Real-time status indicators
- Clean, intuitive UI

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### Step 2: Verify Model File

Ensure the `yolov8s_custom.pt` model file is in the project directory.

## Running the Application

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage Guide

### 1. Upload Image Mode
1. Click on the "📷 Upload Image" tab
2. Click "Browse files" and select an image (JPG, JPEG, or PNG)
3. View the original image and detection results side by side
4. Check the PPE status indicators below

### 2. Upload Video Mode
1. Click on the "🎥 Upload Video" tab
2. Upload a video file (MP4, AVI, or MOV)
3. Click "Start Video Processing"
4. Watch real-time detection as the video plays
5. View PPE status updates for each frame

### 3. Live Webcam Mode
1. Click on the "📹 Live Webcam" tab
2. Click "Start Webcam" button
3. Allow camera access if prompted
4. View real-time PPE detection from your webcam
5. Click "Stop Webcam" to end the session

## Understanding the Results

### Color Coding
- **🟢 Green boxes**: Helmets detected
- **🔵 Blue boxes**: Gloves detected
- **🔴 Red boxes**: Safety shoes detected
- **🟡 Cyan boxes**: Safety vests detected
- **🟣 Magenta boxes**: Safety glasses detected

### Status Indicators
- **✅ Present**: PPE item detected with confidence
- **❌ Not Present**: PPE item not found in frame

## System Requirements

- Python 3.8 or higher
- Webcam (for live detection mode)
- Minimum 4GB RAM
- GPU recommended for faster processing (optional)

## Troubleshooting

### Webcam Not Working
- Ensure your webcam is properly connected
- Check if other applications are using the camera
- Grant camera permissions to your browser/application

### Model File Not Found
- Verify `yolov8s_custom.pt` exists in the project directory
- Check the file path in `streamlit_app.py`

### Slow Performance
- Reduce video resolution
- Process fewer frames (adjust `process_every_n_frames` in code)
- Use a GPU if available

## Technical Details

- **Model**: YOLOv8s (Small) - Custom trained
- **Framework**: Ultralytics YOLO
- **UI Framework**: Streamlit
- **Computer Vision**: OpenCV
- **Image Processing**: PIL/Pillow

## Performance Notes

- **Images**: Instant processing
- **Videos**: Processes every 5th frame by default for optimal performance
- **Webcam**: Processes every 2nd frame for real-time experience

## Customization

### Adjust Detection Frequency
In `streamlit_app.py`, modify:
- `process_every_n_frames` for video processing
- Frame skip logic in webcam detection

### Add More PPE Classes
Update the `ppe_classes` dictionary in `detect_ppe_in_frame()` function

### Change Color Scheme
Modify the `colors` dictionary in `detect_ppe_in_frame()` function

## License

This project uses the YOLOv8 model from Ultralytics under the AGPL-3.0 license.

## Support

For issues or questions, please check:
1. All dependencies are correctly installed
2. Model file is in the correct location
3. Python version is 3.8 or higher
