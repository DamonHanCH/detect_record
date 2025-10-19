import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO
import torch
from flask import Flask, Response, render_template_string
import threading
import numpy as np
import json

# Flask app initialization
app = Flask(__name__)
# Global variables for two cameras' latest frames
latest_frame1 = None
latest_frame2 = None
lock1 = threading.Lock()
lock2 = threading.Lock()

# Video stream page template with two cameras
HTML_TEMPLATE = """
<html>
<head>
    <title>Dual Camera Monitoring</title>
    <style>
        body { 
            display: flex; 
            justify-content: center; 
            align-items: flex-start; 
            padding: 20px; 
            background: #f0f0f0; 
            flex-wrap: wrap;
            gap: 20px;
        }
        .video-container { 
            border: 5px solid #333; 
            border-radius: 10px; 
            padding: 10px;
            background: white;
        }
    </style>
</head>
<body>
    <div class="video-container">
        <h2>Camera 1</h2>
        <img src="/video_feed1" style="max-width: 100%; height: auto;">
    </div>
    <div class="video-container">
        <h2>Camera 2</h2>
        <img src="/video_feed2" style="max-width: 100%; height: auto;">
    </div>
</body>
</html>
"""

def enhance_image(image):
    denoised = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    # Slight sharpening
    kernel = np.array([[0, -0.1, 0],
                       [-0.1, 1.4, -0.1],
                       [0, -0.1, 0]])
    frame = cv2.filter2D(denoised, -1, kernel)
    
    return frame

# Video feed generators for two cameras
def generate_frames1():
    global latest_frame1, lock1
    while True:
        with lock1:
            if latest_frame1 is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame1)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames2():
    global latest_frame2, lock2
    while True:
        with lock2:
            if latest_frame2 is None:
                continue
            ret, buffer = cv2.imencode('.jpg', latest_frame2)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route definitions
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_frames1(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    print(torch.cuda.is_available())
    print(torch.__version__)

    # Load configuration
    with open('config.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Configuration parameters
    DETECTION_INTERVAL = config.get("DETECTION_INTERVAL", 25)
    STOP_CONSECUTIVE_NO_DETECT = config.get("STOP_CONSECUTIVE_NO_DETECT", 20)
    TARGET_CLASS_ID = config.get("TARGET_CLASS_ID")
    RECORD_DIR = config.get("RECORD_DIR", "recordings")
    FIXED_FPS = config.get("FIXED_FPS", 25)

    # Create save directory
    os.makedirs(RECORD_DIR, exist_ok=True)

    # Load YOLO model
    model = YOLO(config.get("MODEL_PATH"))  # 确保config中有MODEL_PATH配置
    class_names = model.names
    for class_id, class_name in class_names.items():
        print(f"ID: {class_id} → name: {class_name}")

    # Initialize two cameras
    cap1 = cv2.VideoCapture(config.get("Capture_index1", 0))
    cap2 = cv2.VideoCapture(config.get("Capture_index2", 1))

    # Check camera connections
    if not cap1.isOpened():
        print("Failed to open camera 1!")
        return
    if not cap2.isOpened():
        print("Failed to open camera 2!")
        cap1.release()
        return

    # Set camera resolutions
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("Capture_WIDTH1", 1280))
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("Capture_HEIGHT1", 720))
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("Capture_WIDTH2", 1280))
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("Capture_HEIGHT2", 720))

    # Get actual resolutions
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera 1 resolution: {width1}x{height1}")
    print(f"Camera 2 resolution: {width2}x{height2}")
    print(f"Recording frame rate: {FIXED_FPS} FPS")
    print(f"Detection interval: once every {DETECTION_INTERVAL} frames")
    print(f"Stop condition: no target detected for {STOP_CONSECUTIVE_NO_DETECT} consecutive times")

    # Recording control variables (shared between cameras)
    is_recording = False
    consecutive_no_detection = 0
    out1 = None  # Video writers for two cameras
    out2 = None
    start_datetime = None
    temp_video_path1 = None  # Temporary file paths
    temp_video_path2 = None
    frame_counter = 0
    last_detection1 = False  # Last detection results for each camera
    last_detection2 = False

    # Start Flask server
    def run_flask():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Flask server started, access http://IP:5000 to view both cameras")

    try:
        while True:
            # Read frames from both cameras
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                print("Failed to get frames from cameras!")
                break

            # Enhance images
            # frame1 = enhance_image(frame1)
            # frame2 = enhance_image(frame2)

            frame_counter += 1
            detected1 = False
            detected2 = False
            annotated_frame1 = frame1.copy()
            annotated_frame2 = frame2.copy()

            # Detection every DETECTION_INTERVAL frames
            if frame_counter % DETECTION_INTERVAL == 0:
                current_time = datetime.now()
                millisecond = current_time.microsecond // 1000
                timestamp_str = f"{current_time.year}-{current_time.month:02d}-{current_time.day:02d} " \
                                f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.{millisecond:03d}"
                print("Current timestamp:", timestamp_str)

                # Detect for both cameras
                results1 = model(frame1, conf=0.3, imgsz=960)
                results2 = model(frame2, conf=0.3, imgsz=960)

                detected1 = any(int(box.cls) == TARGET_CLASS_ID for result in results1 for box in result.boxes)
                detected2 = any(int(box.cls) == TARGET_CLASS_ID for result in results2 for box in result.boxes)

                # Update last detection results
                last_detection1 = detected1
                last_detection2 = detected2

                # Draw annotations
                annotated_frame1 = results1[0].plot()
                annotated_frame2 = results2[0].plot()
            else:
                # Use last detection results
                detected1 = last_detection1
                detected2 = last_detection2

            # Combined detection status (any camera detects target)
            combined_detected = detected1 or detected2

            # Update recording status
            if frame_counter % DETECTION_INTERVAL == 0:
                if combined_detected:
                    consecutive_no_detection = 0
                    if not is_recording:
                        # Start recording for both cameras
                        start_datetime = datetime.now()
                        start_str = start_datetime.strftime("%Y%m%d_%H%M%S")
                        
                        # Create video writers for both cameras
                        temp_video_path1 = os.path.join(RECORD_DIR, f"temp_cam1_{start_str}.avi")
                        temp_video_path2 = os.path.join(RECORD_DIR, f"temp_cam2_{start_str}.avi")
                        
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out1 = cv2.VideoWriter(temp_video_path1, fourcc, FIXED_FPS, (width1, height1))
                        out2 = cv2.VideoWriter(temp_video_path2, fourcc, FIXED_FPS, (width2, height2))
                        
                        if out1.isOpened() and out2.isOpened():
                            is_recording = True
                            print(f"Start recording both cameras at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                        else:
                            print("Failed to create video writers!")
                            out1 = None
                            out2 = None
                            temp_video_path1 = None
                            temp_video_path2 = None
                else:
                    if is_recording:
                        consecutive_no_detection += 1
                        print(f"Consecutive no-detection: {consecutive_no_detection}/{STOP_CONSECUTIVE_NO_DETECT}")

            # Handle recording
            if is_recording and out1 and out2:
                out1.write(frame1)
                out2.write(frame2)

                # Stop recording condition
                if consecutive_no_detection >= STOP_CONSECUTIVE_NO_DETECT:
                    end_datetime = datetime.now()
                    end_str = end_datetime.strftime("%Y%m%d_%H%M%S")
                    
                    # Release and rename both files
                    out1.release()
                    out2.release()
                    
                    final1 = os.path.join(RECORD_DIR, f"recording_cam1_{start_str}_{end_str}.avi")
                    final2 = os.path.join(RECORD_DIR, f"recording_cam2_{start_str}_{end_str}.avi")
                    
                    os.rename(temp_video_path1, final1)
                    os.rename(temp_video_path2, final2)
                    
                    is_recording = False
                    consecutive_no_detection = 0
                    print(f"Stop recording both cameras at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Saved: {final1} and {final2}")

            # Update latest frames for web streaming
            with lock1:
                global latest_frame1
                latest_frame1 = annotated_frame1
            with lock2:
                global latest_frame2
                latest_frame2 = annotated_frame2

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup resources
        if is_recording and out1 and out2:
            out1.release()
            out2.release()
            if temp_video_path1 and os.path.exists(temp_video_path1):
                end_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                final1 = os.path.join(RECORD_DIR, f"recording_cam1_{start_str}_{end_str}_interrupted.avi")
                os.rename(temp_video_path1, final1)
            if temp_video_path2 and os.path.exists(temp_video_path2):
                final2 = os.path.join(RECORD_DIR, f"recording_cam2_{start_str}_{end_str}_interrupted.avi")
                os.rename(temp_video_path2, final2)
            print("Program interrupted, recordings saved")

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        print("Program exited")


if __name__ == "__main__":
    main()