from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import winsound
import math
import time
import threading
import csv
import os
global prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
font = cv2.HISTCMP_BHATTACHARYYA
font_scale = 1
font_thickness =2
yolo_model = YOLO('C:\Desktop\object-tracking-yolov8-native-main\yolov8n.pt')

prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y = 0, 0, 0, 0, 0, 0, 0, 0
start_time = None
min_x = 0
min_y = 0
max_x = 0
max_y = 0

# Create a folder to save the frames
frame_folder = "captured_frames"
if not os.path.exists(frame_folder):
    os.makedirs(frame_folder)

green_box_present = False
first_output = True

def start_timer():
    global start_time
    start_time = time.time()

def record_time(event, elapsed_time, frame):
    global min_x, min_y, max_x, max_y
    global first_output
    # Skip processing the first output
    if first_output:
        first_output = False
        return

    statement = f"{event} detected. Time stamp: {elapsed_time:.2f} seconds"
    print(statement)

    csv_file = "timestamp.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([statement])

    if green_box_present:
        # Draw the rectangle on the frame
        cv2.rectangle(frame, (int(min_x * frame.shape[1]), int(min_y * frame.shape[0])),
                      (int(max_x * frame.shape[1]), int(max_y * frame.shape[0])), (0, 255, 0), 2)

    frame_filename = f"{frame_folder}/{event}_frame_{time.time()}.png"
    cv2.imwrite(frame_filename, frame)
    print(f"Frame saved: {frame_filename}")

def detect_running_pose(pose_landmarks):
    left_knee = pose_landmarks[mp_holistic.PoseLandmark.LEFT_KNEE]
    right_knee = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_KNEE]
    left_ankle = pose_landmarks[mp_holistic.PoseLandmark.LEFT_ANKLE]
    right_ankle = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_ANKLE]

    left_leg_angle = math.degrees(math.atan2(left_ankle.y - left_knee.y, left_ankle.x - left_knee.x))
    right_leg_angle = math.degrees(math.atan2(right_ankle.y - right_knee.y, right_ankle.x - right_knee.x))

    running_threshold = 100

    return left_leg_angle > running_threshold and right_leg_angle > running_threshold

def detect_human_actions(frame):
    global prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y
    global min_x, min_y, max_x, max_y
    global alert_text, green_box_present

    # Detect humans using YOLO
    results = yolo_model(frame)
    
    for obj in results.xyxy[results.names.index('person')]:
        min_x, min_y, max_x, max_y = obj[:4]
        min_x, min_y, max_x, max_y = int(min_x * frame.shape[1]), int(min_y * frame.shape[0]), int(max_x * frame.shape[1]), int(max_y * frame.shape[0])
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        green_box_present = True

        # Perform anomaly detection based on the detected human bounding box
        pose_landmarks = mp_holistic.Holistic().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks
        if pose_landmarks:
            # Additional code for anomaly detection using pose landmarks
            left_knee = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE.value]
            right_knee = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE.value]

            left_leg_angle = math.degrees(math.atan2(left_ankle.y - left_knee.y, left_ankle.x - left_knee.x))
            right_leg_angle = math.degrees(math.atan2(right_ankle.y - right_knee.y, right_ankle.x - right_knee.x))

            jump_threshold = 60

            if abs(left_leg_angle) > jump_threshold and abs(right_leg_angle) > jump_threshold:
                elapsed_time = time.time() - start_time
                record_time("Jumping", elapsed_time, frame)
                duration = 1000
                freq = 440
                winsound.Beep(freq, duration)

                cv2.putText(frame, 'JUMPING', (50, 50), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            prev_left_ear_x, prev_left_ear_y = left_ear_x, left_ear_y
            prev_right_ear_x, prev_right_ear_y = right_ear_x, right_ear_y
            prev_left_shoulder_x, prev_left_shoulder_y = left_shoulder_x, left_shoulder_y
            prev_right_shoulder_x, prev_right_shoulder_y = right_shoulder_x, right_shoulder_y

            if detect_running_pose(pose_landmarks):
                elapsed_time = time.time() - start_time
                record_time("Running", elapsed_time, frame)
                duration = 1000
                freq = 1000
                winsound.Beep(freq, duration)

                cv2.putText(frame, 'RUNNING', (50, 50), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            if pose_landmarks:
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])

                mean_position = np.mean(landmarks_array[:, :2], axis=0)
                distances = np.linalg.norm(landmarks_array[:, :2] - mean_position, axis=1)
                all_on_same_line = np.all(distances < 0.1)

                if all_on_same_line:
                    elapsed_time = time.time() - start_time
                    record_time("Crawling", elapsed_time, frame)
                    duration = 1000
                    freq = 600
                    winsound.Beep(freq, duration)

                    cv2.putText(frame, 'CRAWLING', (50, 50), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

# The rest of the code remains unchanged
def show_frame(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.rectangle(frame, (int(min_x * frame.shape[1]), int(min_y * frame.shape[0])),(int(max_x * frame.shape[1]), int(max_y * frame.shape[0])), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('K'):
            break
    cap.release()
    cv2.destroyAllWindows()



def run_inference(cap):
    while True:
        start_timer()
        ret, frame = cap.read()
        if not ret:
            break
        detect_human_actions(frame)

        if cv2.waitKey(1) & 0xFF == ord('K'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    t1 = threading.Thread(target=show_frame, args=(cap,), name='t1')
    t2 = threading.Thread(target=run_inference, args=(cap,), name='t2')
    # starting threads
    t1.start()
    t2.start()

    # wait until all threads finish
    t1.join()
    t2.join()
