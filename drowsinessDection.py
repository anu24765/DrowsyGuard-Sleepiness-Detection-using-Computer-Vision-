import cv2
import dlib
import numpy as np
import pygame
from imutils import face_utils
# Load pre-trained face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define eye landmarks indices
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Define function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    eye = np.array(eye, dtype=np.int32)
    p1 = np.linalg.norm(eye[1] - eye[5])
    p2 = np.linalg.norm(eye[2] - eye[4])
    p3 = np.linalg.norm(eye[0] - eye[3])
    return (p1 + p2) / (2.0 * p3)

# Initialize drowsiness-related variables
EYE_AR_THRESH = 0.3  # Threshold for eye aspect ratio
EYE_AR_CONSEC_FRAMES = 28  # Consecutive frames for drowsiness detection
COUNTER = 0  # Frame counter for consecutive closed eyes

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  # Load alarm sound

# Create a resizable window and set it to full screen
cv2.namedWindow('Drowsiness Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Drowsiness Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 0)

    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract eye landmarks
        left_eye = shape[lstart:lend]
        right_eye = shape[rstart:rend]

        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the eye aspect ratios
        ear = (left_ear + right_ear) / 2.0

        # Drowsiness detection logic
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)  # Play alarm continuously
        else:
            COUNTER = 0
            pygame.mixer.music.stop()  # Stop alarm

        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            cv2.putText(frame, "Drowsiness Detected!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw eye regions on the frame
        left_hull = cv2.convexHull(left_eye)
        right_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_hull], -1, (0, 255, 0), 1)

    # Get the screen resolution
    screen_res = cv2.getWindowProperty('Drowsiness Detection', cv2.WND_PROP_FULLSCREEN)

    # Display the resulting frame in the full screen window
    cv2.imshow('Drowsiness Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()