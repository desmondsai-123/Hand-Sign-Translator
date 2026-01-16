import cv2
import av
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import pickle

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Neural View Stable", layout="wide")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        # Using a cached function ensures the model only loads once
        model_dict = pickle.load(open('./model.p', 'rb'))
        return model_dict['model']
    except FileNotFoundError:
        st.error("Error: 'model.p' not found.")
        return None

model = load_model()

# --- 3. STABLE PROCESSOR CLASS ---
class ModernProcessor(VideoTransformerBase):
    def __init__(self):
        # Initializing hands here instead of recv prevents freezing
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.5,
            model_complexity=0 # Lightest model for stability
        )
        self.generated_text = ""
        self.last_char = ""
        self.prev = ""
        self.frames = 0
        self.labels_list = {i: chr(65+i) for i in range(26)} # A-Z mapping

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h_img, w_img, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux, x_, y_ = [], [], []
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw Landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 122, 255), thickness=1)
            )
            
            for lm in hand_landmarks.landmark:
                x_.append(lm.x); y_.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))

            if model:
                # Pad to 84 features for model compatibility
                features = np.asarray(data_aux[:84])
                if len(features) < 84:
                    features = np.pad(features, (0, 84 - len(features)))

                try:
                    prediction = model.predict([features])[0]
                    char = self.labels_list.get(int(prediction), "?")

                    # Lock-in Logic
                    if char == self.prev: 
                        self.frames += 1
                    else: 
                        self.frames = 0; self.prev = char

                    if self.frames == 12: 
                        if char != self.last_char:
                            self.generated_text += char
                            self.last_char = char
                        self.frames = 0

                    cv2.putText(img, char, (int(min(x_)*w_img), int(min(y_)*h_img)-20), 
                                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 122, 255), 2)
                except:
                    pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. MAIN INTERFACE ---
st.markdown("<h3 style='text-align: center;'>Neural View Translation</h3>", unsafe_allow_html=True)

col_video, col_results = st.columns([3, 2]) 

with col_video:
    ctx = webrtc_streamer(
        key="neural-view",
        video_processor_factory=ModernProcessor,
        async_processing=True, # Critical for preventing UI freeze
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_results:
    st.markdown("### 📝 Translated Text")
    
    # Show text box and clear button
    current_text = ctx.video_processor.generated_text if ctx.video_processor else ""
    st.markdown(f"""<div style='background:#f0f2f6; padding:20px; border-radius:10px; border-left:5px solid #007aff; min-height:200px; color:#333;'>
                {current_text if current_text else "<i>Waiting for gesture...</i>"}</div>""", unsafe_allow_html=True)
    
    if st.button("✨ Clear Text", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.generated_text = ""
            st.rerun() # Clean way to refresh UI
