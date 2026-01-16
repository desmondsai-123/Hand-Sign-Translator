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
st.set_page_config(page_title="Neural Sign Translator", layout="wide")

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
        self.current_char = ""    
        self.generated_text = "" 
        self.prev = ""           
        self.frames = 0          
        self.cooldown = False    

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
                    self.current_char = char

                    # --- 35 FRAME LOGIC ---
                    if char == self.prev:
                        if not self.cooldown:
                            self.frames += 1
                    else:
                        self.frames = 0
                        self.prev = char
                        self.cooldown = False 

                    # Updated to exactly 35 frames
                    if self.frames >= 35:
                        self.generated_text += char
                        self.frames = 0
                        self.cooldown = True 
                    
                    # Visual progress bar on the video feed
                    display_color = (0, 255, 0) if self.cooldown else (0, 165, 255)
                    cv2.putText(img, f"Stability: {self.frames}/35", (10, 450), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)
                except: pass
        else:
            self.current_char = ""
            self.frames = 0 
            self.cooldown = False 

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. INTERFACE ---
st.title("🤟 Neural Sign Translator")
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
    st.subheader("Detected Sign")
    char_spot = st.empty()
    
    st.markdown("---")
    st.subheader("Built Word")
    text_box = st.empty() 
    
    if st.button("🗑️ Clear Word", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.generated_text = ""
        st.rerun()

# --- 4. THE UI UPDATE LOOP ---
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            curr = ctx.video_processor.current_char
            char_spot.markdown(f"<h1 style='text-align: center; color: #007aff; font-size: 80px;'>{curr if curr else '-'}</h1>", unsafe_allow_html=True)
            
            full_text = ctx.video_processor.generated_text
            # Using a large font for the final word box
            text_box.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #007aff;'>
                    <h2 style='color: #31333F; margin: 0;'>{full_text if full_text else "..."}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        if not ctx.state.playing:
            break
        time.sleep(0.1)
