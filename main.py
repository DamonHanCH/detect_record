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

# Added: Flask app initialization
app = Flask(__name__)
# Global variable to store the latest frame
latest_frame = None
lock = threading.Lock()

# Added: Video stream page template
HTML_TEMPLATE = """
<html>
<head>
    <title>Monitoring Screen</title>
    <style>
        body { display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; background: #f0f0f0; }
        .video-container { border: 5px solid #333; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="video-container">
        <h1>Monitoring Screen</h1>
        <img src="/video_feed" style="max-width: 100%; height: auto;">
    </div>
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
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
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

    # Create save directory
    os.makedirs(RECORD_DIR, exist_ok=True)

    # Load YOLOv8-world model
    model = YOLO(config.get(".pt_MODEL"))

    # Open camera
    cap = cv2.VideoCapture(config.get("Capture_index"))
    if not cap.isOpened():
        print("Failed to open camera, please check the device!")
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

    # Added: Start Flask server thread
    def run_flask():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Flask server started, you can access http://computer-IP:5000 via LAN on your phone to view the screen")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to get image frame, exiting program")
                break

            frame_counter += 1
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
                results = model(frame, conf=0.3)
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
                        temp_video_path = os.path.join(RECORD_DIR, f"temp_{start_str}.avi")
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
                    final_video_path = os.path.join(RECORD_DIR, final_video_name)
                    os.rename(temp_video_path, final_video_path)
                    is_recording = False
                    consecutive_no_detection = 0
                    print(f"Stop recording (end time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')})")
                    print(f"Recording saved as: {final_video_name}")

            # Added: Update the latest frame for mobile viewing
            with lock:
                global latest_frame
                latest_frame = annotated_frame

            # Local display
            # cv2.imshow("YOLOv8-world Detection & Recording", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if is_recording and out is not None:
            out.release()
            if temp_video_path and os.path.exists(temp_video_path):
                end_datetime = datetime.now()
                end_str = end_datetime.strftime("%Y%m%d_%H%M%S")
                final_video_name = f"recording_{start_datetime.strftime('%Y%m%d_%H%M%S')}_{end_str}_interrupted.avi"
                final_video_path = os.path.join(RECORD_DIR, final_video_name)
                os.rename(temp_video_path, final_video_path)
                print(f"Program forced to exit, recording saved as: {final_video_name}")
        cap.release()
        cv2.destroyAllWindows()
        print("Program exited")


if __name__ == "__main__":
    main()