import cv2
import av
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import pickle
import time

# --- 1. SETUP ---
st.set_page_config(page_title="VisionAI Sign Translator", layout="wide")

@st.cache_resource
def load_vision_model():
    try:
        with open('./model.p', 'rb') as f:
            return pickle.load(f)['model']
    except:
        return None

model = load_vision_model()

# --- 2. THE PROCESSOR ---
class SignProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            model_complexity=0 
        )
        self.labels = {i: chr(65+i) for i in range(26)}
        self.current_char = "" # This stores the last prediction
        self.prev = ""
        self.frames = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            data_aux = []
            x_ = [lm.x for lm in hand_lms.landmark]
            y_ = [lm.y for lm in hand_lms.landmark]
            for lm in hand_lms.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if model:
                input_data = np.asarray(data_aux[:84])
                if len(input_data) < 84:
                    input_data = np.pad(input_data, (0, 84 - len(input_data)))
                
                try:
                    prediction = model.predict([input_data])[0]
                    char = self.labels.get(int(prediction), "?")
                    
                    # Update the attribute directly for the main script to read
                    self.current_char = char
                    cv2.putText(img, char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except: pass
        else:
            self.current_char = ""

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. THE INTERFACE ---
st.title("Sign Language Translator")
col_vid, col_txt = st.columns([2, 1])

with col_vid:
    # We assign this to 'ctx' to access the processor later
    ctx = webrtc_streamer(
        key="sign-translate",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignProcessor,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_txt:
    st.subheader("Detected Sign")
    # A placeholder that we will update in the loop below
    char_placeholder = st.empty()
    
    if st.button("Clear Screen"):
        st.rerun()

# --- 4. THE UI UPDATE LOOP ---
# This loop runs while the camera is active and reads data from the processor
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            # Read the current_char from the background thread
            char = ctx.video_processor.current_char
            if char:
                char_placeholder.markdown(f"<h1 style='text-align: center; color: #007aff; font-size: 100px;'>{char}</h1>", unsafe_allow_html=True)
            else:
                char_placeholder.info("Show a hand sign to start...")
        
        # Stop the loop if the user clicks 'STOP' on the camera
        if not ctx.state.playing:
            break
            
        time.sleep(0.1) # Prevents CPU usage from spiking
