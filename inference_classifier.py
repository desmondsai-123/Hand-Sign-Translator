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

# --- LABELS (FIXED NUMBERING 0-49) ---
labels_list = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 
    19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z_0", 26: "Z_1", 
    27: "How are you?", 
    28: "Waalaikumussalam", 
    29: "Hello", 
    30: "I'm fine", 
    31: "Excuse me", 
    32: "Sorry", 
    33: "Salam", 
    34: "Regards", 
    35: "You're welcome", 
    36: "Well", 
    37: "Come", 
    38: "Birthday", 
    39: "Goodbye", 
    40: "Night", 
    41: "Morning", 
    42: "Please (Welcome)", 
    43: "Thank you", 
    44: "Please (Help)", 
}

# --- 1. HIDDEN SIGNS (Silent Context) ---
# Signs that don't print but set up a context for the next sign.
hidden_signs = [
    "I don't want to renumber everything so this is hidden",
    "How are you?",
]

# --- 2. MODIFIERS (Strict Blocking) ---
# These words CANNOT exist alone. They MUST follow their prerequisite.
modifiers = {
    "J": "I",
    "Please (Welcome)": "Goodbye"
}

# --- 3. ACTIVATORS (Merging / Optional Combination) ---
# These words CAN exist alone. 
# BUT, if they follow a specific word, they merge into a new word.
# Format: { ("Previous_Word", "Current_Word"): "Resulting_Output" }
activators = {
    # If "Hello" is on screen, and you sign "Waalaikumussalam", it becomes "Assalamualaikum"
    ("Hello", "Waalaikumussalam"): "Assalamualaikum",
    ("Well", "Morning"): "Good Morning",
    ("Well", "Night"): "Good Night",
    ("Well", "Birthday"): "Happy Birthday",
    ("Well", "Come"): "Welcome",
    ("How are you?", "I'm fine"): "How are you?",
}

# --- CONFIGURATION ---
probability_threshold = 0.5
stability_threshold = 10     
reset_threshold = 10         

# --- STATE VARIABLES ---
previous_prediction = None    
consecutive_frames = 0        
nothing_consecutive_frames = 0 

# Variables for the "Sequence" logic
last_printed_char = None      
last_valid_entry = None       
was_last_hidden = False       

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

                # --- 1. SEQUENCE LOGIC (Z_0, Z_1) ---
                if "_" in predicted_label and predicted_label not in hidden_signs: 
                    base_name, step_str = predicted_label.split("_")
                    if step_str.isdigit():
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
                            was_last_hidden = False

                # --- 2. STANDARD LOGIC ---
                else:
                    current_sequence_base = None 
                    sequence_step = -1
                    
                    if predicted_label != last_printed_char:
                        
                        # A. Check Hidden Signs (Context)
                        if predicted_label in hidden_signs:
                            last_valid_entry = predicted_label
                            was_last_hidden = True
                            last_printed_char = predicted_label
                        
                        else:
                            should_print = True
                            is_modifier_replace = False
                            
                            # B. Check Strict Modifiers (Blocking)
                            if predicted_label in modifiers:
                                required_prev = modifiers[predicted_label]
                                if last_valid_entry == required_prev:
                                    is_modifier_replace = True
                                else:
                                    should_print = False # Block if requirement not met

                            if should_print:
                                # C. Check Activators (Merging)
                                combo_key = (last_valid_entry, predicted_label)
                                
                                if combo_key in activators:
                                    # Case: MERGE TRIGGERED (e.g. Hello + Waalaik... -> Assalam...)
                                    new_word = activators[combo_key]
                                    
                                    # If previous word was visible (like "Hello"), delete it
                                    if not was_last_hidden:
                                        delete_len = len(last_valid_entry) + 1
                                        print("\b" * delete_len, end="", flush=True)
                                    
                                    # Print the new Merged Word
                                    print(new_word, end=" ", flush=True)
                                    last_valid_entry = new_word
                                    was_last_hidden = False
                                
                                elif is_modifier_replace:
                                    # Case: STRICT MODIFIER (e.g. Well + Birthday -> Happy Birthday)
                                    if not was_last_hidden:
                                        delete_len = len(last_valid_entry) + 1
                                        print("\b" * delete_len, end="", flush=True)
                                    
                                    print(predicted_label, end=" ", flush=True)
                                    last_valid_entry = predicted_label
                                    was_last_hidden = False

                                else:
                                    # Case: NORMAL PRINT
                                    print(predicted_label, end=" ", flush=True)
                                    last_valid_entry = predicted_label
                                    was_last_hidden = False
                                
                                last_printed_char = predicted_label

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
            # We preserve last_valid_entry to allow combos across pauses

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
