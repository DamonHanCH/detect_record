[INTRODUCTION]
Based on a personal computer and a camera, it uses a detection model to identify the category of interest, automatically records and saves it as a local video, and simultaneously sends real-time images to the local area network (LAN) for easy viewing.
yolov8l-world.pt download from https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-world.pt

[config]
{
  "RECORD_DIR": Path for saving recorded videos; it will be created automatically if it does not exist
  ".pt_MODEL":  Path to the model file
  "TARGET_CLASS_ID": Class ID of the target category of interest in the model
  "Capture_index": Device number of the camera
  "Capture_WIDTH": Width of the image captured by the camera
  "Capture_HEIGHT": Height of the image captured by the camera
  "DETECTION_INTERVAL": Number of frames between each detection; detection cannot be performed on every frame as it would take too much time
  "STOP_CONSECUTIVE_NO_DETECT": Stop recording if the target of interest is not detected in consecutive detections
  "FIXED_FPS": Frame rate of the recording
  "end": Unused
}

[RUN]
#install requirements
	please execute command: pip install -r requirements.txt
#config
	please edit config.json
#run
	please execute command: python main.py