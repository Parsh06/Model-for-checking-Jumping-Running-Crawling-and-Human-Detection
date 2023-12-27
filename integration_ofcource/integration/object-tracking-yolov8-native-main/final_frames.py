import cv2
import mediapipe as mp
import numpy as np
# import winsound
import math
import time
import threading
import csv
import os 

# org = (50, 50)  # Example coordinates (x, y)
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# color = (255, 0, 0)  # Example color in BGR format (green in this case)
# font_thickness = 2
# lineType = cv2.LINE_AA
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
prev_left_ear_x, prev_left_ear_y, prev_right_ear_x, prev_right_ear_y, prev_left_shoulder_x, prev_left_shoulder_y, prev_right_shoulder_x, prev_right_shoulder_y = 0, 0, 0, 0, 0, 0, 0, 0
start_time = None
pTime = 0
cTime = 0
min_x = 0
min_y = 0
max_x = 0
max_y = 0
frame=None
# Create a folder to save the frames
# frame_folder = "captured_frames"
# if not os.path.exists(frame_folder):
#     os.makedirs(frame_folder)

# green_box_present = False
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

    # csv_file = "timestamp.csv"
    # with open(csv_file, mode='a', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([statement])

    # if green_box_present:
    #     # Draw the rectangle on the frame
    #     cv2.rectangle(frame, (int(min_x * frame.shape[1]), int(min_y * frame.shape[0])),(int(max_x * frame.shape[1]), int(max_y * frame.shape[0])), (0, 255, 0), 2)

    # frame_filename = f"{frame_folder}/{event}_frame_{time.time()}.png"
    # cv2.imwrite(frame_filename, frame)
    # print(f"Frame saved: {frame_filename}")

# def record_time(event, elapsed_time, frame):
#     statement = (f"{event} detected. Time stamp: {elapsed_time:.2f} seconds")
#     print(statement)

#     csv_file = "timestamp.csv"
#     with open(csv_file, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([statement])

#     if green_box_present:
#         frame_filename = f"{frame_folder}/{event}_frame_{time.time()}.png"
#         cv2.imwrite(frame_filename, frame)
#         print(f"Frame saved: {frame_filename}")

#     frame_filename = f"{frame_folder}/{event}_frame_{time.time()}.png"
#     cv2.imwrite(frame_filename, frame)
#     print(f"Frame saved: {frame_filename}")

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
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # green_box_present = False

    if results.pose_landmarks:
        min_x = min([lm.x for lm in results.pose_landmarks.landmark])
        min_y = min([lm.y for lm in results.pose_landmarks.landmark])
        max_x = max([lm.x for lm in results.pose_landmarks.landmark])
        max_y = max([lm.y for lm in results.pose_landmarks.landmark])

        # cv2.rectangle(image, (int(min_x * image.shape[1]), int(min_y * image.shape[0])),(int(max_x * image.shape[1]), int(max_y * image.shape[0])), (0, 255, 0), 2)
        # green_box_present = True

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark


        # nose = pose_landmarks[mp_holistic.PoseLandmark.NOSE]
        left_shoulder = pose_landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        left_ear = pose_landmarks[mp_holistic.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks[mp_holistic.PoseLandmark.RIGHT_EAR]

        left_ear_x, left_ear_y = int(left_ear.x * image.shape[1]), int(left_ear.y * image.shape[0])
        right_ear_x, right_ear_y = int(right_ear.x * image.shape[1]), int(right_ear.y * image.shape[0])
        left_shoulder_x, left_shoulder_y = int(left_shoulder.x * image.shape[1]), int(left_shoulder.y * image.shape[0])
        right_shoulder_x, right_shoulder_y = int(right_shoulder.x * image.shape[1]), int(right_shoulder.y * image.shape[0])

        # left_ear_dx = abs(left_ear_x - prev_left_ear_x)
        left_ear_dy = abs(left_ear_y - prev_left_ear_y)
        # right_ear_dx = abs(right_ear_x - prev_right_ear_x)
        right_ear_dy = abs(right_ear_y - prev_right_ear_y)
        # left_shoulder_dx = abs(left_shoulder_x - prev_left_shoulder_x)
        left_shoulder_dy = abs(left_shoulder_y - prev_left_shoulder_y)
        # right_shoulder_dx = abs(right_shoulder_x - prev_right_shoulder_x)
        right_shoulder_dy = abs(right_shoulder_y - prev_right_shoulder_y)

        jump_threshold = 60

        if left_ear_dy > jump_threshold and right_ear_dy > jump_threshold and left_shoulder_dy > jump_threshold and right_shoulder_dy > jump_threshold:
            # elapsed_time = time.time() - start_time
            # record_time("Jumping", elapsed_time, frame)
            print("jump")
            #cv2.putText(frame, 'Jumping', org, font, fontScale, color, thickness, lineType)
            duration = 1000
            freq = 440
            # winsound.Beep(freq, duration)
            text_position = (int(frame.shape[1] / 10), int(frame.shape[0] / 10))
            cv2.putText(frame, 'JUMPING', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4, cv2.LINE_AA)

        prev_left_ear_x, prev_left_ear_y = left_ear_x, left_ear_y
        prev_right_ear_x, prev_right_ear_y = right_ear_x, right_ear_y
        prev_left_shoulder_x, prev_left_shoulder_y = left_shoulder_x, left_shoulder_y
        prev_right_shoulder_x, prev_right_shoulder_y = right_shoulder_x, right_shoulder_y

        if detect_running_pose(pose_landmarks):
            # elapsed_time = time.time() - start_time
            # record_time("Running", elapsed_time, frame)
            print("run")
            duration = 1000
            freq = 1000
            # winsound.Beep(freq, duration)

        if results.pose_landmarks is not None:
            pose_landmarks = results.pose_landmarks
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark])

            mean_position = np.mean(landmarks_array[:, :2], axis=0)
            distances = np.linalg.norm(landmarks_array[:, :2] - mean_position, axis=1)
            all_on_same_line = np.all(distances < 0.1)

            if all_on_same_line:
                # elapsed_time = time.time() - start_time
                # record_time("Crawling", elapsed_time, frame)
                print("crawl")
                duration = 1000

                freq = 600
                # winsound.Beep(freq, duration)


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
