import cv2
import time
import os
import argparse
from ultralytics import YOLO

def record_on_detection(
    model_path="yolov8l-world.pt",
    output_dir="recordings",
    camera_index=0,
    resolution=(1920, 1080),
    base_duration=20,  # 基础录制时长（秒）
    target_fps=30,
    conf_threshold=0.5
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Videos will be saved to: {os.path.abspath(output_dir)}")
    
    # 加载模型
    try:
        model = YOLO(model_path)
        print(f"Successfully loaded model: {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 配置摄像头（优化缓冲区和帧率）
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 只缓存最新帧
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    # 获取实际参数并适配帧率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_camera_fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = min(target_fps, int(actual_camera_fps) if actual_camera_fps > 0 else target_fps)
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print(f"Using target FPS: {target_fps} (camera actual: {actual_camera_fps:.1f})")
    print(f"Recording strategy: 20s base, reset timer if cat re-detected")

    # 状态变量
    is_recording = False  # 是否正在录制
    out = None  # 视频写入对象
    start_time_str = ""  # 录制开始时间戳（用于文件名）
    last_detection_time = 0  # 上次检测时间（控制1秒1次检测）
    cat_detected = False  # 当前是否检测到猫
    results = None  # 缓存检测结果
    remaining_time = 0.0  # 剩余录制时间（核心：动态倒计时）
    last_frame_time = time.time()  # 上一帧的时间（用于计算时间差）

    print("Starting monitoring... (Press 'q' to exit)")

    while True:
        # 1. 优先读取帧（最高优先级）
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame, exiting")
            break

        # 计算当前帧与上一帧的时间差（用于更新倒计时）
        current_time = time.time()
        time_elapsed = current_time - last_frame_time
        last_frame_time = current_time

        # 2. 检测逻辑（1秒1次，控制开销）
        if current_time - last_detection_time >= 1.0:
            last_detection_time = current_time
            results = model(frame, classes=[15], conf=conf_threshold, imgsz=480, verbose=False)
            cat_detected = len(results[0].boxes) > 0
            # 若检测到猫，重置剩余录制时间为20秒
            if cat_detected:
                remaining_time = base_duration  # 核心：重置倒计时
                print(f"Cat detected at {time.strftime('%H:%M:%S')} - reset timer to 20s")

        # 3. 录制逻辑（基于动态倒计时）
        # 更新剩余时间（每帧都减，确保精度）
        if remaining_time > 0:
            remaining_time -= time_elapsed
            # 若需要开始录制（从非录制状态进入录制）
            if not is_recording:
                is_recording = True
                start_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(current_time))
                temp_file = os.path.join(output_dir, f"temp_{start_time_str}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_file, fourcc, target_fps, (actual_width, actual_height))
                print(f"Start recording (remaining: {remaining_time:.1f}s)")
            # 写入当前帧（录制中）
            out.write(frame)
        # 若倒计时结束且正在录制，停止并保存
        elif is_recording:
            is_recording = False
            out.release()
            end_time_str = time.strftime("%H%M%S", time.localtime(current_time))
            final_file = os.path.join(output_dir, f"{start_time_str}-{end_time_str}.mp4")
            os.rename(temp_file, final_file)
            print(f"Recording saved to: {final_file} (total duration: {base_duration + (start_time_str != end_time_str)*0:.1f}s)")

        # 4. 显示逻辑（低优先级）
        display_frame = frame.copy()
        # 非录制时绘制检测框（录制时专注写入）
        if not is_recording and cat_detected and results is not None:
            display_frame = results[0].plot()
        # 显示剩余录制时间（若正在录制）
        if is_recording:
            cv2.putText(display_frame, f"Recording - remaining: {max(0, remaining_time):.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            status = "Cat detected (will record 20s)" if cat_detected else "No cat"
            cv2.putText(display_frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if cat_detected else (0, 0, 255), 2)
        cv2.imshow("Monitoring (q to exit)", display_frame)

        # 退出逻辑
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 资源清理
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Program exited")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cat detection recording (reset timer on re-detection)")
    parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("-o", "--output", type=str, default="recordings", help="Output directory")
    parser.add_argument("-d", "--duration", type=int, default=20, help="Base recording duration (s, default:20)")
    parser.add_argument("-f", "--fps", type=int, default=30, help="Target FPS")
    args = parser.parse_args()

    record_on_detection(
        camera_index=args.camera,
        output_dir=args.output,
        base_duration=args.duration,
        target_fps=args.fps
    )