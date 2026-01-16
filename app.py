import cv2
import av
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
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
        st.error(f"Error loading model: {e}")
        return None

model = load_vision_model()

# --- 2. THE PROCESSOR (Merged from inference_classifier.py) ---
class SignProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            model_complexity=0 
        )
        self.labels_list = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 
            10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 
            19: "T", 20: "U", 21: "V", 22: "W", 23: "X", 24: "Y", 25: "Z_0", 26: "Z_1", 
            27: "How are you?", 28: "Waalaikumussalam", 29: "Hello", 30: "I'm fine", 
            31: "Excuse me", 32: "Sorry", 33: "Salam", 34: "Regards", 35: "You're welcome", 
            36: "Well", 37: "Come", 38: "Birthday", 39: "Goodbye", 40: "Night", 
            41: "Morning", 42: "Please (Welcome)", 43: "Thank you", 44: "Please (Help)", 
        }
        self.activators = {
            ("Hello", "Waalaikumussalam"): "Assalamualaikum",
            ("Well", "Morning"): "Good Morning",
            ("Well", "Night"): "Good Night",
            ("Well", "Birthday"): "Happy Birthday",
            ("Well", "Come"): "Welcome",
            ("How are you?", "I'm fine"): "How are you?",
        }
        self.generated_text = ""
        self.current_char = ""
        self.last_valid_entry = None
        self.consecutive_frames = 0
        self.previous_prediction = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks and model:
            hand_lms = results.multi_hand_landmarks[0]
            data_aux = []
            x_ = [lm.x for lm in hand_lms.landmark]
            y_ = [lm.y for lm in hand_lms.landmark]
            for lm in hand_lms.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Logic Parity with inference_classifier.py
            if len(data_aux) < 84: data_aux.extend([0.0] * (84 - len(data_aux)))
            prediction_proba = model.predict_proba([np.asarray(data_aux)])
            confidence = np.max(prediction_proba)

            if confidence > 0.5:
                idx = int(np.argmax(prediction_proba))
                label = self.labels_list.get(idx, "?")
                self.current_char = label

                if label == self.previous_prediction:
                    self.consecutive_frames += 1
                else:
                    self.consecutive_frames = 0
                    self.previous_prediction = label

                if self.consecutive_frames == 10: # stability_threshold
                    combo_key = (self.last_valid_entry, label)
                    if combo_key in self.activators:
                        # Merge Logic
                        words = self.generated_text.strip().split(" ")
                        if words: words.pop()
                        self.generated_text = " ".join(words) + " " + self.activators[combo_key]
                    else:
                        self.generated_text += " " + label
                    self.last_valid_entry = label
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. THE INTERFACE ---
st.title("🤟 Neural Sign Word Builder")
col_vid, col_txt = st.columns([2, 1])

with col_vid:
    ctx = webrtc_streamer(
        key="sign-translate",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignProcessor,
        async_processing=False, # Crucial for stability on Python 3.12
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_txt:
    st.subheader("Current Sign")
    char_spot = st.empty()
    st.subheader("Sentence")
    text_box = st.empty()

# --- 4. SAFE UI UPDATE ---
if ctx.state.playing:
    while ctx.state.playing:
        if ctx.video_processor:
            char_spot.info(f"Detected: {ctx.video_processor.current_char}")
            text_box.success(ctx.video_processor.generated_text)
        time.sleep(0.1)
