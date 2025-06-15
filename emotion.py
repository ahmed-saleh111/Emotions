import cv2
from ultralytics import YOLO

# Load YOLO emotion detection model
model_face = YOLO("models\yolov11n-face.pt") 
model_emotion = YOLO("models\yolo11s-emotion.pt") 

# Define emotion labels
emotion_labels = {
    0: 'anger', 1: 'content', 2: 'disgust', 3: 'fear',
    4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'
}
# emotion_labels = model_emotion.names

def enlarge_box(box, scale_factor, img_width, img_height):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    new_x1 = max(0, x1 - width * scale_factor)
    new_y1 = max(0, y1 - height * scale_factor)
    new_x2 = min(img_width, x2 + width * scale_factor)
    new_y2 = min(img_height, y2 + height * scale_factor)
    
    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

def process_emotion_frame(frame, scale_factor=0.2):
    """
    Detects faces and emotions in the given frame, annotates the frame with bounding boxes and emotion labels.
    
    Parameters:
        frame (np.ndarray): The input image frame.
        scale_factor (float): How much to enlarge the bounding box when cropping faces.
    
    Returns:
        np.ndarray: The annotated frame.
    """
    img_height, img_width = frame.shape[:2]

    # 1) Face Detection
    results = model_face(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        enlarged_box = enlarge_box(box, scale_factor, img_width, img_height)
        x1, y1, x2, y2 = enlarged_box
        face_crop = frame[y1:y2, x1:x2]  # face 

        # 2) Run emotion detection on cropped face
        emotion_result = model_emotion(face_crop, conf=0.75)

        # Get emotion class
        emotion_cls = emotion_result[0].boxes.cls #tensor([2])
        if emotion_cls.numel() == 0:
            continue  # No emotion detected

        cls_id = int(emotion_cls[0].item())  # 2
        label = emotion_labels.get(cls_id, "unknown")

        # Draw bounding box on the original frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

    return frame


def main():
    cap = cv2.VideoCapture(r"videos\video4.mp4")  # Open the default webcam (0 means primary camera)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit")

    cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Emotion Recognition", 1500, 1200)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process face recognition
        processed_frame = process_emotion_frame(frame)

        # Display the processed frame
        cv2.imshow("Emotion Recognition", processed_frame)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


