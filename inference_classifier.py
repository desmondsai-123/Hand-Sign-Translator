import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings

# Suppress protobuf warning
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# --- LABELS ---
labels_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# --- PREREQUISITE RULES ---
# Format: { "Target": "Required_Previous" }
modifiers = {
    "J": "I"   # J will REPLACE I
}

# --- CONFIGURATION ---
probability_threshold = 0.7 
stability_threshold = 10     
reset_threshold = 10         

# --- STATE VARIABLES ---
previous_prediction = None    
consecutive_frames = 0        
nothing_consecutive_frames = 0 

# Variables for the "Sequence" logic
last_printed_char = None      
last_valid_entry = None       

current_sequence_base = None  
sequence_step = -1            

print("System Ready. Type away...", flush=True)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break 

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    sign_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)
                
            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

        expected_features = 84
        if len(data_aux) < expected_features:
            data_aux.extend([0.0] * (expected_features - len(data_aux)))

        prediction_proba = model.predict_proba([np.asarray(data_aux)])
        confidence = np.max(prediction_proba)

        # --- VALID DETECTION BLOCK ---
        if confidence > probability_threshold:
            sign_detected = True
            nothing_consecutive_frames = 0 
            
            predicted_label = labels_list[int(np.argmax(prediction_proba))]
            
            # Stability Check
            if predicted_label == previous_prediction:
                consecutive_frames += 1
            else:
                consecutive_frames = 0
                previous_prediction = predicted_label
            
            # Trigger Logic
            if consecutive_frames == stability_threshold:
                
                # CASE 1: Sequence Parts
                if "_" in predicted_label:
                    base_name, step_str = predicted_label.split("_")
                    step = int(step_str)

                    if step == 0:
                        current_sequence_base = base_name
                        sequence_step = 0
                    
                    elif step == 1 and current_sequence_base == base_name and sequence_step == 0:
                        print(base_name, end=" ", flush=True)
                        sequence_step = -1 
                        current_sequence_base = None
                        last_printed_char = base_name 
                        last_valid_entry = base_name 

                # CASE 2: Normal Characters
                else:
                    current_sequence_base = None 
                    sequence_step = -1
                    
                    if predicted_label != last_printed_char:
                        
                        # --- PREREQUISITE SYSTEM START ---
                        allowed_to_print = True
                        is_replacement = False # New flag

                        if predicted_label in modifiers:
                            required_prev = modifiers[predicted_label]
                            
                            if last_valid_entry != required_prev:
                                allowed_to_print = False
                            else:
                                is_replacement = True # It matched, so we enable replacement mode
                        # --- PREREQUISITE SYSTEM END ---

                        if allowed_to_print:
                            # If this is a replacement (J replacing I), backspace the old one
                            if is_replacement:
                                # Calculate length to delete (Letter length + 1 for the space we added)
                                delete_len = len(last_valid_entry) + 1
                                print("\b" * delete_len, end="", flush=True)

                            print(predicted_label, end=" ", flush=True)
                            last_printed_char = predicted_label
                            last_valid_entry = predicted_label 

            # Visual Feedback
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f"{predicted_label} {int(confidence*100)}%", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # --- NO HANDS / LOW CONFIDENCE BLOCK ---
    if not sign_detected:
        nothing_consecutive_frames += 1
        consecutive_frames = 0 
        
        if nothing_consecutive_frames > reset_threshold:
            last_printed_char = None
            # Do NOT reset last_valid_entry

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()