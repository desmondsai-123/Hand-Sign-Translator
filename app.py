import cv2
import av
import numpy as np
import mediapipe as mp
# Direct imports to bypass AttributeError on Linux/Cloud environments
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import pickle
import queue

# --- 1. SETUP ---
st.set_page_config(page_title="VisionAI Stable", layout="wide")

if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if "text_out" not in st.session_state:
    st.session_state.text_out = ""

# --- 2. MODEL LOAD ---
@st.cache_resource
def load_vision_model():
    try:
        with open('./model.p', 'rb') as f:
            return pickle.load(f)['model']
    except:
        return None

model = load_vision_model()

# --- 3. VIDEO ENGINE ---
class StableProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            model_complexity=0 
        )
        self.labels = {i: chr(65+i) for i in range(26)}
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
                # Ensure exactly 84 features
                input_data = np.asarray(data_aux[:84])
                if len(input_data) < 84:
                    input_data = np.pad(input_data, (0, 84 - len(input_data)))
                
                try:
                    prediction = model.predict([input_data])[0]
                    char = self.labels.get(int(prediction), "?")

                    if char == self.prev:
                        self.frames += 1
                    else:
                        self.frames = 0
                        self.prev = char

                    if self.frames == 15: # Lock in after 15 frames
                        st.session_state.result_queue.put(char)
                        self.frames = 0
                    
                    cv2.putText(img, char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except: pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. INTERFACE ---
st.title("Sign Language Translator")
col1, col2 = st.columns([3, 2])

with col1:
    webrtc_streamer(
        key="stable-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=StableProcessor,
        async_processing=True, # Critical for preventing freeze
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# Transfer from background thread to UI thread
while not st.session_state.result_queue.empty():
    st.session_state.text_out += st.session_state.result_queue.get()

with col2:
    st.subheader("Transcription")
    st.info(st.session_state.text_out if st.session_state.text_out else "Awaiting gestures...")
    if st.button("Clear"):
        st.session_state.text_out = ""
        st.rerun()
