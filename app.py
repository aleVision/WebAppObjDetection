import streamlit as st
import torch
import cv2
import numpy as np

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Object classes for COCO dataset
COCO_CLASSES = {
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    1: 'bicycle'
}

# Map selected objects to class IDs
DETECTABLE_OBJECTS = {
    'Pedestrian (People)': 0,
    'Car': 2,
    'Motorcycle': 3,
    'Bus': 5,
    'Truck': 7,
    'Bicycle': 1
}

# Function to perform detection and filtering based on selected objects
def detect_objects(frame, model, selected_objects):
    results = model(frame)
    detected_objects = results.pandas().xyxy[0]  # Get bounding boxes and results in pandas DataFrame

    # Filter the detected objects based on user's selection
    class_ids = [DETECTABLE_OBJECTS[obj] for obj in selected_objects]  # Map selected objects to class IDs
    filtered_objects = detected_objects[detected_objects['class'].isin(class_ids)]

    return filtered_objects

# Draw bounding boxes around detected objects
def draw_boxes(frame, detections):
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        class_id = int(row['class'])
        label = f"{COCO_CLASSES[class_id]}: {row['confidence']:.2f}"
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

# Streamlit app configuration
st.title("Real-Time Object Detection and Tracking with YOLOv5")

# Allow the user to select which objects to detect
selected_objects = st.multiselect(
    'Choose the objects you want to detect and track:',
    list(DETECTABLE_OBJECTS.keys()),
    default=['Pedestrian (People)', 'Car']
)

# Video file uploader
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if not selected_objects:
    st.warning("Please select at least one object to detect.")
elif uploaded_video is None:
    st.warning("Please upload a video file to start detection.")
else:
    # Load the YOLOv5 model
    model = load_model()

    # Load video file using OpenCV
    video_file = uploaded_video.name
    video_bytes = uploaded_video.read()
    with open(video_file, 'wb') as f:
        f.write(video_bytes)
    
    cap = cv2.VideoCapture(video_file)

    # Display real-time video feed with detections
    stframe = st.empty()  # Placeholder for video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Video processing complete.")
            break

        # Detect objects based on the selected categories
        detected_objects = detect_objects(frame, model, selected_objects)

        # Draw bounding boxes on the detected objects
        frame_with_boxes = draw_boxes(frame, detected_objects)

        # Display the video frame with detections
        stframe.image(frame_with_boxes, channels="BGR", use_column_width=True)

    cap.release()
