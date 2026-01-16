import cv2
import av
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import pickle
import gzip
import time

# --- 1. SETUP ---
st.set_page_config(page_title="Neural Sign Translator", layout="wide")

@st.cache_resource
def load_vision_model():
    model_path = './model_compressed.p.gz' 
    try:
        with gzip.open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
            return model_dict['model']
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure 'model_compressed.p.gz' is in your GitHub repo.")
        return None

model = load_vision_model()

# --- 2. THE PROCESSOR ---
class SignProcessor:
    def __init__(self):
        # MediaPipe Setup
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            model_complexity=0 
        )
        
        # --- LOGIC CONFIG FROM INFERENCE_CLASSIFIER.PY ---
        self.labels_list = {
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

        self.hidden_signs = [
            "I don't want to renumber everything so this is hidden",
            "How are you?",
        ]

        self.modifiers = {
            "J": "I",
            "Please (Welcome)": "Goodbye"
        }

        self.activators = {
            ("Hello", "Waalaikumussalam"): "Assalamualaikum",
            ("Well", "Morning"): "Good Morning",
            ("Well", "Night"): "Good Night",
            ("Well", "Birthday"): "Happy Birthday",
            ("Well", "Come"): "Welcome",
            ("How are you?", "I'm fine"): "How are you?",
        }

        # Configuration
        self.probability_threshold = 0.5
        self.stability_threshold = 10     
        self.reset_threshold = 10         

        # State Variables
        self.current_char = ""    
        self.generated_text = "" 
        self.previous_prediction = None
        self.consecutive_frames = 0
        self.nothing_consecutive_frames = 0
        
        # Sequence Logic State
        self.last_printed_char = None
        self.last_valid_entry = None
        self.was_last_hidden = False
        self.current_sequence_base = None
        self.sequence_step = -1

    def update_text(self, new_text, is_replace=False, is_merge=False):
        """Helper to update the generated_text string similar to how 'print' works"""
        current_words = self.generated_text.strip().split(" ")
        
        if is_merge or is_replace:
            if current_words:
                current_words.pop() # Remove the last word (Simulates \b)
        
        if new_text:
            current_words.append(new_text)
            
        self.generated_text = " ".join(current_words)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        sign_detected = False
        display_text = "Listening..."

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                # Draw Landmarks
                mp_drawing.draw_landmarks(
                    img, 
                    hand_lms, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract Data
                data_aux = []
                x_ = [lm.x for lm in hand_lms.landmark]
                y_ = [lm.y for lm in hand_lms.landmark]
                
                for lm in hand_lms.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                # Draw Bounding Box (Visual Parity)
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)

                if model:
                    # Pad data to expected 84 features
                    if len(data_aux) < 84:
                        data_aux.extend([0.0] * (84 - len(data_aux)))
                    
                    try:
                        # Use predict_proba for confidence check
                        prediction_proba = model.predict_proba([np.asarray(data_aux)])
                        confidence = np.max(prediction_proba)
                        
                        if confidence > self.probability_threshold:
                            sign_detected = True
                            self.nothing_consecutive_frames = 0
                            
                            predicted_index = int(np.argmax(prediction_proba))
                            predicted_label = self.labels_list.get(predicted_index, "?")
                            self.current_char = predicted_label # Update UI big letter
                            
                            display_text = f"{predicted_label} {int(confidence*100)}%"
                            
                            # --- STABILITY LOGIC ---
                            if predicted_label == self.previous_prediction:
                                self.consecutive_frames += 1
                            else:
                                self.consecutive_frames = 0
                                self.previous_prediction = predicted_label
                                
                            # --- TRIGGER LOGIC ---
                            if self.consecutive_frames == self.stability_threshold:
                                
                                # 1. SEQUENCE LOGIC (Z_0 -> Z_1)
                                if "_" in predicted_label and predicted_label not in self.hidden_signs:
                                    base_name, step_str = predicted_label.split("_")
                                    if step_str.isdigit():
                                        step = int(step_str)
                                        if step == 0:
                                            self.current_sequence_base = base_name
                                            self.sequence_step = 0
                                        elif step == 1 and self.current_sequence_base == base_name and self.sequence_step == 0:
                                            # Sequence Complete: Print Base Name
                                            self.update_text(base_name)
                                            self.sequence_step = -1
                                            self.current_sequence_base = None
                                            self.last_printed_char = base_name
                                            self.last_valid_entry = base_name
                                            self.was_last_hidden = False
                                
                                # 2. STANDARD LOGIC
                                else:
                                    self.current_sequence_base = None
                                    self.sequence_step = -1
                                    
                                    if predicted_label != self.last_printed_char:
                                        
                                        # A. Hidden Signs
                                        if predicted_label in self.hidden_signs:
                                            self.last_valid_entry = predicted_label
                                            self.was_last_hidden = True
                                            self.last_printed_char = predicted_label
                                        
                                        else:
                                            should_print = True
                                            is_modifier_replace = False
                                            
                                            # B. Modifiers (Strict)
                                            if predicted_label in self.modifiers:
                                                required_prev = self.modifiers[predicted_label]
                                                if self.last_valid_entry == required_prev:
                                                    is_modifier_replace = True
                                                else:
                                                    should_print = False
                                            
                                            if should_print:
                                                # C. Activators (Merging)
                                                combo_key = (self.last_valid_entry, predicted_label)
                                                
                                                if combo_key in self.activators:
                                                    new_word = self.activators[combo_key]
                                                    
                                                    # If previous was visible, remove it first
                                                    is_merge = not self.was_last_hidden
                                                    self.update_text(new_word, is_merge=is_merge)
                                                    
                                                    self.last_valid_entry = new_word
                                                    self.was_last_hidden = False
                                                    
                                                elif is_modifier_replace:
                                                    is_replace = not self.was_last_hidden
                                                    self.update_text(predicted_label, is_replace=is_replace)
                                                    
                                                    self.last_valid_entry = predicted_label
                                                    self.was_last_hidden = False
                                                
                                                else:
                                                    # Normal Print
                                                    self.update_text(predicted_label)
                                                    self.last_valid_entry = predicted_label
                                                    self.was_last_hidden = False
                                                
                                                self.last_printed_char = predicted_label

                            cv2.putText(img, display_text, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Prediction Error: {e}")
                        pass
        
        # --- NO HANDS LOGIC ---
        if not sign_detected:
            self.nothing_consecutive_frames += 1
            self.consecutive_frames = 0
            self.current_char = "" # Clear big letter
            
            if self.nothing_consecutive_frames > self.reset_threshold:
                self.last_printed_char = None
                # self.last_valid_entry preserved for pauses

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. THE INTERFACE ---
st.title("🤟 Neural Sign Word Builder")
col_vid, col_txt = st.columns([2, 1])

with col_vid:
    ctx = webrtc_streamer(
        key="sign-translate",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_txt:
    st.subheader("Current Detected Sign")
    char_spot = st.empty()
    
    st.markdown("---")
    st.subheader("Built Sentence / Word")
    text_box = st.empty() 
    
    if st.button("🗑️ Clear Everything", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.generated_text = ""
            ctx.video_processor.last_valid_entry = None
            ctx.video_processor.last_printed_char = None
        st.rerun()

# --- 4. THE UI UPDATE LOOP ---
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            # 1. Update the Big Letter
            curr = ctx.video_processor.current_char
            char_spot.markdown(f"<h1 style='text-align: center; color: #007aff; font-size: 100px;'>{curr if curr else '-'}</h1>", unsafe_allow_html=True)
            
            # 2. Update the Word Box
            full_text = ctx.video_processor.generated_text
            text_box.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 25px; border-radius: 10px; border-left: 8px solid #007aff;'>
                    <h2 style='color: #31333F; margin: 0; font-family: monospace;'>{full_text if full_text else "Start signing..."}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        if not ctx.state.playing:
            break
        time.sleep(0.1)
