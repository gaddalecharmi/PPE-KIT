import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="PPE Detection System",
    page_icon="🦺",
    layout="wide"
)

# Load YOLO model
@st.cache_resource
def load_model():
    """Load the YOLOv8 model"""
    # Using the custom model from the project
    model_path = "yolov8s_custom.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        return model
    else:
        st.error(f"Model file '{model_path}' not found. Please ensure the model file exists.")
        return None

def detect_ppe_in_frame(frame, model):
    """
    Run PPE detection on a single frame
    Returns: annotated frame, detection results
    """
    results = model(frame, verbose=False)
    
    # Define PPE classes we're looking for
    ppe_classes = {
        'helmet': ['Helmet', 'helmet'],
        'gloves': ['Gloves', 'gloves'],
        'shoes': ['Shoes', 'shoes', 'Safety-Shoes'],
        'vest': ['Safety-Vest', 'vest'],
        'glasses': ['Glass', 'glasses', 'Safety-Glasses']
    }
    
    detected_items = {
        'helmet': False,
        'gloves': False,
        'shoes': False,
        'vest': False,
        'glasses': False
    }
    
    # Draw bounding boxes and collect detections
    annotated_frame = frame.copy()
    
    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            
            # Check which PPE category this detection belongs to
            for ppe_type, class_variants in ppe_classes.items():
                if class_name in class_variants:
                    detected_items[ppe_type] = True
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Different colors for different PPE types
                    colors = {
                        'helmet': (0, 255, 0),      # Green
                        'gloves': (255, 0, 0),      # Blue
                        'shoes': (0, 0, 255),       # Red
                        'vest': (255, 255, 0),      # Cyan
                        'glasses': (255, 0, 255)    # Magenta
                    }
                    
                    color = colors.get(ppe_type, (255, 255, 255))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame, detected_items

def display_ppe_status(detected_items):
    """Display PPE detection status in a nice format"""
    st.subheader("PPE Detection Results:")
    
    cols = st.columns(5)
    
    ppe_info = [
        ("Helmet", "helmet", "🪖"),
        ("Gloves", "gloves", "🧤"),
        ("Shoes", "shoes", "👟"),
        ("Safety Vest", "vest", "🦺"),
        ("Glasses", "glasses", "🥽")
    ]
    
    for col, (name, key, emoji) in zip(cols, ppe_info):
        with col:
            status = "✅ Present" if detected_items[key] else "❌ Not Present"
            color = "green" if detected_items[key] else "red"
            st.markdown(f"### {emoji} {name}")
            st.markdown(f"<p style='color: {color}; font-size: 18px; font-weight: bold;'>{status}</p>", 
                       unsafe_allow_html=True)

def process_image(image, model):
    """Process uploaded image"""
    # Convert PIL image to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:  # RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Run detection
    annotated_frame, detected_items = detect_ppe_in_frame(img_array, model)
    
    # Convert back to RGB for display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame, detected_items

def process_video(video_path, model):
    """Process uploaded video"""
    cap = cv2.VideoCapture(video_path)
    
    stframe = st.empty()
    status_placeholder = st.empty()
    
    frame_count = 0
    process_every_n_frames = 5  # Process every 5th frame for performance
    
    # Persistent detection status - once detected, stays detected
    persistent_status = {
        'helmet': False,
        'gloves': False,
        'shoes': False,
        'vest': False,
        'glasses': False
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % process_every_n_frames == 0:
            # Run detection
            annotated_frame, detected_items = detect_ppe_in_frame(frame, model)
            
            # Update persistent status - once True, stays True
            for key in persistent_status:
                if detected_items[key]:
                    persistent_status[key] = True
            
            # Convert BGR to RGB for display
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            # Display persistent status
            with status_placeholder.container():
                display_ppe_status(persistent_status)
    
    cap.release()
    st.success("Video processing complete!")

def run_webcam_detection(model):
    """Run real-time webcam detection"""
    st.info("Starting webcam... Press 'Stop Webcam' button to stop.")
    
    # Create stop button
    stop_button = st.button("Stop Webcam", key="stop_webcam")
    
    # Placeholders for video and status
    stframe = st.empty()
    status_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not access webcam. Please check your camera connection.")
        return
    
    frame_count = 0
    
    # Persistent detection status - once detected, stays detected
    persistent_status = {
        'helmet': False,
        'gloves': False,
        'shoes': False,
        'vest': False,
        'glasses': False
    }
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam")
            break
        
        frame_count += 1
        
        # Process every frame for real-time detection
        if frame_count % 2 == 0:  # Process every other frame for performance
            annotated_frame, detected_items = detect_ppe_in_frame(frame, model)
            
            # Update persistent status - once True, stays True
            for key in persistent_status:
                if detected_items[key]:
                    persistent_status[key] = True
            
            # Convert BGR to RGB for display
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            stframe.image(annotated_frame, channels="RGB", use_container_width=True)
            
            # Display persistent status
            with status_placeholder.container():
                display_ppe_status(persistent_status)
        
        # Check if stop button was pressed
        if stop_button:
            break
    
    cap.release()
    st.success("Webcam stopped.")

def main():
    st.title("🦺 PPE Detection System")
    st.markdown("### Personal Protective Equipment Detection using YOLOv8")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Create tabs for different input modes
    tab1, tab2, tab3 = st.tabs(["📷 Upload Image", "🎥 Upload Video", "📹 Live Webcam"])
    
    # Tab 1: Image Upload
    with tab1:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image file", 
                                        type=['jpg', 'jpeg', 'png'],
                                        key="image_uploader")
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
            
            # Process image
            with st.spinner("Detecting PPE..."):
                annotated_image, detected_items = process_image(image, model)
            
            with col2:
                st.subheader("Detection Results")
                st.image(annotated_image, use_container_width=True)
            
            st.markdown("---")
            display_ppe_status(detected_items)
    
    # Tab 2: Video Upload
    with tab2:
        st.header("Upload a Video")
        uploaded_video = st.file_uploader("Choose a video file", 
                                         type=['mp4', 'avi', 'mov'],
                                         key="video_uploader")
        
        if uploaded_video is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            if st.button("Start Video Processing"):
                with st.spinner("Processing video..."):
                    process_video(tfile.name, model)
                
                # Clean up temporary file
                os.unlink(tfile.name)
    
    # Tab 3: Webcam
    with tab3:
        st.header("Live Webcam Detection")
        st.markdown("Click the button below to start real-time PPE detection from your webcam.")
        
        if st.button("Start Webcam", key="start_webcam"):
            run_webcam_detection(model)
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This PPE Detection System uses YOLOv8 to detect:
        - 🪖 **Helmets**
        - 🧤 **Gloves**
        - 👟 **Safety Shoes**
        - 🦺 **Safety Vests**
        - 🥽 **Safety Glasses**
        
        ### How to Use:
        1. Choose an input mode (Image/Video/Webcam)
        2. Upload your file or start webcam
        3. View detection results with bounding boxes
        4. Check PPE compliance status
        
        ### Color Coding:
        - 🟢 Green: Helmet
        - 🔵 Blue: Gloves
        - 🔴 Red: Shoes
        - 🟡 Cyan: Safety Vest
        - 🟣 Magenta: Glasses
        """)
        
        st.markdown("---")
        st.markdown("**Model:** YOLOv8s Custom")
        st.markdown("**Framework:** Ultralytics")

if __name__ == "__main__":
    main()
