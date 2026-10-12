import cv2
import dlib
import numpy as np
import pyautogui
import threading

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get screen size
screen_width, screen_height = pyautogui.size()

# Webcam capture in a separate thread
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduce frame size
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame = None
running = True


def capture_frames():
    global frame, running
    while running:
        ret, new_frame = cap.read()
        if ret:
            frame = new_frame


thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()


def get_eye_regions(landmarks):
    """Extracts left and right eye regions as NumPy arrays."""
    left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], np.int32)
    right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], np.int32)
    return left_eye_pts, right_eye_pts


def get_pupil_position(eye_frame):
    """Detects pupil inside the eye using adaptive + OTSU thresholding."""
    gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive + OTSU thresholding
    blur = cv2.GaussianBlur(gray_eye, (5, 5), 0)
    _, threshold_eye = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        return x + w // 2, y + h // 2  # Pupil center

    return None


face_box = None  # Store last detected face
while True:
    if frame is None:
        continue

    small_frame = cv2.resize(frame, (320, 240))  # Reduce processing load
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Detect face every 10 frames
    if face_box is None or cv2.waitKey(1) % 10 == 0:
        faces = detector(gray)
        if faces:
            face_box = faces[0]

    if face_box:
        landmarks = predictor(gray, face_box)

        # Get both eye regions at once
        left_eye_pts, right_eye_pts = get_eye_regions(landmarks)

        # Extract eye regions from frame
        min_x = min(left_eye_pts[:, 0].min(), right_eye_pts[:, 0].min())
        max_x = max(left_eye_pts[:, 0].max(), right_eye_pts[:, 0].max())
        min_y = min(left_eye_pts[:, 1].min(), right_eye_pts[:, 1].min())
        max_y = max(left_eye_pts[:, 1].max(), right_eye_pts[:, 1].max())

        eye_frame = frame[min_y:max_y, min_x:max_x].copy()

        # Get both pupil positions in the same cycle
        left_pupil, right_pupil = None, None
        if eye_frame.size > 0:
            left_pupil = get_pupil_position(eye_frame)
            right_pupil = get_pupil_position(eye_frame)

        screen_x, screen_y = screen_width // 2, screen_height // 2  # Default to center

        if left_pupil and right_pupil:
            norm_x = (left_pupil[0] + right_pupil[0]) / 2
            norm_y = (left_pupil[1] + right_pupil[1]) / 2

            # Map to screen coordinates
            screen_x = int((norm_x / eye_frame.shape[1]) * screen_width)
            screen_y = int((norm_y / eye_frame.shape[0]) * screen_height)

            # Move mouse to estimated gaze location (optional)
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)  # Smoother movement

            # Draw pupils
            cv2.circle(frame, (min_x + left_pupil[0], min_y + left_pupil[1]), 3, (0, 255, 0), -1)
            cv2.circle(frame, (min_x + right_pupil[0], min_y + right_pupil[1]), 3, (0, 255, 0), -1)

        cv2.putText(frame, f"Gaze: {screen_x}, {screen_y}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Fast Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break

cap.release()
cv2.destroyAllWindows()







