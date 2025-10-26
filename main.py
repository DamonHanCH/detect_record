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
import argparse


# Added: Flask app initialization
app = Flask(__name__)
# Global variable to store the latest frame
latest_frame = None
lock = threading.Lock()

# Added: FPS statistics
fps_stats = {
    "current_fps": 0,
    "average_fps": 0,
    "frame_count": 0,
    "start_time": time.time(),
    "fps_history": []  # Store recent FPS values for smoothing
}
fps_lock = threading.Lock()

# Added: Video stream page template with FPS display
HTML_TEMPLATE = """
<html>
<head>
    <title>Monitoring Screen - Camera {{ camera_index }}</title>
    <style>
        body { display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #f0f0f0; }
        .video-container { border: 5px solid #333; border-radius: 10px; padding: 20px; background: white; }
        .fps-info { margin-top: 10px; padding: 10px; background: #f5f5f5; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="video-container">
        <h1>Monitoring Screen - Camera {{ camera_index }}</h1>
        <img src="/video_feed" style="max-width: 100%; height: auto;">
        <div class="fps-info">
            <h3>Frame Rate Statistics</h3>
            <p>Current FPS: <span id="current-fps">0</span></p>
            <p>Average FPS: <span id="average-fps">0</span></p>
            <p>Total Frames: <span id="total-frames">0</span></p>
        </div>
    </div>
    
    <script>
        function updateFPS() {
            fetch('/fps_stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('current-fps').textContent = data.current_fps.toFixed(2);
                    document.getElementById('average-fps').textContent = data.average_fps.toFixed(2);
                    document.getElementById('total-frames').textContent = data.frame_count;
                })
                .catch(error => console.error('Error fetching FPS data:', error));
        }
        
        // Update FPS every second
        setInterval(updateFPS, 1000);
        // Initial update
        updateFPS();
    </script>
</body>
</html>
"""


# Added: Generate video stream
def generate_frames():
    global latest_frame, lock
    while True:
        with lock:
            if latest_frame is None:
                continue
            # Convert to JPEG format
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()

        # Transmit in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Added: Route definition
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, camera_index=camera_index)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Added: FPS statistics route
@app.route('/fps_stats')
def get_fps_stats():
    with fps_lock:
        return {
            "current_fps": fps_stats["current_fps"],
            "average_fps": fps_stats["average_fps"],
            "frame_count": fps_stats["frame_count"]
        }


# Added: Function to update FPS statistics
def update_fps_stats():
    global fps_stats
    with fps_lock:
        current_time = time.time()
        elapsed_time = current_time - fps_stats["start_time"]
        
        # Calculate current FPS (frames per second)
        if elapsed_time > 0:
            current_fps = fps_stats["frame_count"] / elapsed_time
        else:
            current_fps = 0
        
        # Add to history and keep only last 10 values
        fps_stats["fps_history"].append(current_fps)
        if len(fps_stats["fps_history"]) > 10:
            fps_stats["fps_history"].pop(0)
        
        # Calculate smoothed average FPS
        if fps_stats["fps_history"]:
            fps_stats["average_fps"] = sum(fps_stats["fps_history"]) / len(fps_stats["fps_history"])
        
        fps_stats["current_fps"] = current_fps


# Added: Function to draw FPS information on frame
def draw_fps_info(frame, camera_index):
    with fps_lock:
        current_fps = fps_stats["current_fps"]
        average_fps = fps_stats["average_fps"]
        frame_count = fps_stats["frame_count"]
    
    # Create FPS info text
    fps_text = f"Cam{camera_index} - FPS: {current_fps:.1f} (Avg: {average_fps:.1f}) - Frames: {frame_count}"
    
    # Draw background rectangle for better text visibility
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (10, 10), (20 + text_size[0], 40), (0, 0, 0), -1)
    
    # Draw FPS text
    cv2.putText(frame, fps_text, (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame


def main(cap_index, flask_port):
    global camera_index
    camera_index = cap_index  # Make camera_index available globally for the template
    
    print(torch.cuda.is_available())  # Output True means GPU is available
    print(torch.__version__)

    # Open JSON file
    with open('config.json', 'r', encoding='utf-8') as file:
        # Read and parse the file
        config = json.load(file)

    # Macro definitions
    DETECTION_INTERVAL = config.get("DETECTION_INTERVAL")  # Detect once every 25 frames
    STOP_CONSECUTIVE_NO_DETECT = config.get("STOP_CONSECUTIVE_NO_DETECT")  # Stop recording if no detection for 20 consecutive times

    # Configuration parameters
    TARGET_CLASS_ID = config.get("TARGET_CLASS_ID")  # Target class ID
    RECORD_DIR = config.get("RECORD_DIR")  # Recording save directory
    FIXED_FPS = config.get("FIXED_FPS")  # Fixed frame rate

    # Create save directory (include cap_index subdirectory)
    save_dir = os.path.join(RECORD_DIR, str(cap_index))
    os.makedirs(save_dir, exist_ok=True)

    # Load YOLOv8-world model
    model = YOLO(config.get("MODEL_PATH"))
    class_names = model.names

    for class_id, class_name in class_names.items():
        print(f"ID: {class_id} → name: {class_name}")

    # Open camera
    cap = cv2.VideoCapture(cap_index)
    if not cap.isOpened():
        print(f"Failed to open camera (index: {cap_index}), please check the device!")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("Capture_WIDTH"))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("Capture_HEIGHT"))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}, Recording frame rate: {FIXED_FPS} FPS")
    print(f"Detection interval: once every {DETECTION_INTERVAL} frames")
    print(f"Stop condition: no target detected for {STOP_CONSECUTIVE_NO_DETECT} consecutive times")

    # Recording control variables
    is_recording = False
    consecutive_no_detection = 0
    out = None
    start_datetime = None
    temp_video_path = None
    frame_counter = 0
    last_detection_result = False

    # Added: Start Flask server thread with specified port
    def run_flask():
        app.run(host='0.0.0.0', port=flask_port, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"Flask server started (camera index: {cap_index}), access via http://局域网IP:{flask_port}")

    # Added: FPS update thread
    def update_fps_periodically():
        while True:
            update_fps_stats()
            time.sleep(1)  # Update FPS stats every second
    
    fps_thread = threading.Thread(target=update_fps_periodically, daemon=True)
    fps_thread.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get image frame, exiting program")
                break

            frame_counter += 1
            
            # Added: Update frame count for FPS calculation
            with fps_lock:
                fps_stats["frame_count"] += 1
            
            detected = False
            annotated_frame = frame.copy()  # Initialize annotated frame

            # Detect once every DETECTION_INTERVAL frames
            if frame_counter % DETECTION_INTERVAL == 0:
                current_time = datetime.now()
                millisecond = current_time.microsecond // 1000
                timestamp_str = f"{current_time.year}-{current_time.month:02d}-{current_time.day:02d} " \
                                f"{current_time.hour:02d}:{current_time.minute:02d}:{current_time.second:02d}.{millisecond:03d}"
                print("Current detailed timestamp (including milliseconds):", timestamp_str)

                # Object detection
                results = model(frame, conf=0.3, imgsz=960)
                detected = any(int(box.cls) == TARGET_CLASS_ID for result in results for box in result.boxes)
                last_detection_result = detected
                annotated_frame = results[0].plot()  # Draw detection results
            else:
                detected = last_detection_result

            # Update recording status
            if frame_counter % DETECTION_INTERVAL == 0:
                if detected:
                    consecutive_no_detection = 0
                    if not is_recording:
                        start_datetime = datetime.now()
                        start_str = start_datetime.strftime("%Y%m%d_%H%M%S")
                        temp_video_path = os.path.join(save_dir, f"temp_{start_str}.avi")
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(temp_video_path, fourcc, FIXED_FPS, (width, height))
                        if out.isOpened():
                            is_recording = True
                            print(f"Start recording (start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')})")
                        else:
                            print(f"Failed to create recording file: {temp_video_path}")
                            start_datetime = None
                            temp_video_path = None
                else:
                    if is_recording:
                        consecutive_no_detection += 1
                        print(f"Number of consecutive no-detection: {consecutive_no_detection}/{STOP_CONSECUTIVE_NO_DETECT}")

            # Handle recording
            if is_recording:
                out.write(frame)
                if consecutive_no_detection >= STOP_CONSECUTIVE_NO_DETECT:
                    end_datetime = datetime.now()
                    end_str = end_datetime.strftime("%Y%m%d_%H%M%S")
                    out.release()
                    final_video_name = f"recording_{start_datetime.strftime('%Y%m%d_%H%M%S')}_{end_str}.avi"
                    final_video_path = os.path.join(save_dir, final_video_name)
                    os.rename(temp_video_path, final_video_path)
                    is_recording = False
                    consecutive_no_detection = 0
                    print(f"Stop recording (end time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')})")
                    print(f"Recording saved as: {final_video_name}")

            # Added: Draw FPS information on the frame
            display_frame = annotated_frame.copy()
            display_frame = draw_fps_info(display_frame, cap_index)

            # Update the latest frame for mobile viewing
            with lock:
                global latest_frame
                latest_frame = display_frame

            # Local display (commented out by default)
            # cv2.imshow(f"YOLOv8 Detection (Camera {cap_index})", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if is_recording and out is not None:
            out.release()
            if temp_video_path and os.path.exists(temp_video_path):
                end_datetime = datetime.now()
                end_str = end_datetime.strftime("%Y%m%d_%H%M%S")
                final_video_name = f"recording_{start_datetime.strftime('%Y%m%d_%H%M%S')}_{end_str}_interrupted.avi"
                final_video_path = os.path.join(save_dir, final_video_name)
                os.rename(temp_video_path, final_video_path)
                print(f"Program forced to exit, recording saved as: {final_video_name}")
        cap.release()
        cv2.destroyAllWindows()
        print("Program exited")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure camera index and Flask port")
    parser.add_argument("--cap", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--port", type=int, default=5000, help="Flask server port (default: 5000)")
    args = parser.parse_args()
    main(args.cap, args.port)