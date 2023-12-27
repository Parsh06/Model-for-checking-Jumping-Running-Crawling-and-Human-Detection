from ultralytics import YOLO
import cv2
from final_frames import detect_human_actions, show_frame


# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = ''
# cap = cv2.VideoCapture('dog.mp4')
cap = cv2.VideoCapture(0)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        # track objects
        # detect objects
        # results = model.detect(frame)
        results = model.track(frame, persist=True)

        for result in results:
            boxes = result.boxes
            for box in boxes.xyxy:
                x_min, y_min, x_max, y_max = box[:4]
                class_label = box[-1]

                # Draw red bounding box
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

                # Analyze human activities
                detect_human_actions(frame)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
