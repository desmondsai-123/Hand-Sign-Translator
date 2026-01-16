import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# CHANGED: Allow detection of 2 hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Iterate through all directories
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        # Only proceed if at least one hand is detected
        if results.multi_hand_landmarks:
            # First Loop: Get all X and Y values for normalization
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

            # Second Loop: Normalize the values
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # --- CRITICAL CHANGE: PADDING ---
            # We enforce a fixed length of 84 (2 hands * 21 landmarks * 2 coords)
            # If only 1 hand is detected, we pad the rest with zeros.
            expected_length = 84
            if len(data_aux) < expected_length:
                data_aux.extend([0.0] * (expected_length - len(data_aux)))
            
            # (Optional) Safety check: ensure we don't accidentally have more data
            data.append(data_aux[:expected_length])
            labels.append(int(dir_))

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
print("Data processing complete. Saved to data.pickle")