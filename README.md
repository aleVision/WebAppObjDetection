# Real-Time Object Detection and Tracking with YOLOv5 in PyTorch (Video Input)

This project demonstrates real-time object detection and tracking using the YOLOv5 object detection algorithm in PyTorch. Users can upload a video file and select specific objects (pedestrians, cars, trucks, buses, motorcycles, and bicycles) to detect and track.

## 🚀 Live Web App

Check out the live web app here:  
[Object Detection and Tracking Web App](https://your-deployed-app-link.com)

## 🌟 Features

- **Real-Time Detection**: Detect and track objects like pedestrians, cars, trucks, and more from uploaded video files.
- **User Selection**: Choose which objects to detect using an interactive interface. You can track one or multiple objects simultaneously.
- **Bounding Boxes**: Visualize detections with bounding boxes and confidence scores.

## 🛠️ Tools and Technologies

- **YOLOv5 (Ultralytics)**: State-of-the-art object detection model implemented in PyTorch.
- **PyTorch**: Deep learning framework for object detection and classification.
- **OpenCV**: For real-time video processing from uploaded video files.
- **Streamlit**: Web application framework for creating an interactive user interface.

## 📦 How to Run Locally

### Prerequisites

- Python 3.7 or higher
- Git

### Installation Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/ObjectDetectionYOLOv5.git
    cd ObjectDetectionYOLOv5
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the app**:

    ```bash
    streamlit run app.py
    ```

4. **Access the app** in your browser at:

    ```
    http://localhost:8501
    ```

## 🔍 Usage

### Upload a Video

- Upload a video file in **MP4**, **MOV**, or **AVI** format.
- Select the objects you want to detect (e.g., Pedestrians, Cars, Trucks, etc.) from the dropdown menu.
- The app will process the video frame-by-frame and detect the selected objects.

### Object Categories

The app allows you to detect and track the following objects:
- **Pedestrians (People)**
- **Cars**
- **Trucks**
- **Buses**
- **Motorcycles**
- **Bicycles**

## 💡 Future Work

- **Custom Training**: Extend the model by training it on a custom dataset to detect other specific objects.
- **Model Optimization**: Implement model optimization techniques for faster inference on edge devices.
- **Additional Features**: Add functionality for object counting, tracking, or generating alerts when specific objects are detected.

## 📄 References

- **YOLOv5 Repository**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- **PyTorch**: [PyTorch Official Site](https://pytorch.org/)
- **OpenCV**: [OpenCV Documentation](https://docs.opencv.org/)
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io/)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.
