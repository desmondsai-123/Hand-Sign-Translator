import cv2
import av
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import time
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Neural View", layout="wide")

# --- LOAD MODEL ---
# NOTE: Ensure 'model.p' is in the same directory as this script.
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    model = None
    st.error("Error: 'model.p' not found. Please ensure the model file is in the directory.")

# --- PROCESSOR CLASS ---
class ModernProcessor:
    def __init__(self):
        self.generated_text = ""
        self.last_char = ""
        self.prev = ""
        self.frames = 0
        
        # Dictionary for labels (Adjust according to your specific model training)
        self.labels_list = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 
            5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 
            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 
            20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
        }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h_img, w_img, _ = img.shape

        # Initialize MediaPipe inside the thread
        with mp.solutions.hands.Hands(
            static_image_mode=False, 
            max_num_hands=1, 
            min_detection_confidence=0.5
        ) as hands:
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                data_aux, x_, y_ = [], [], []
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw Landmarks (Soft White UI)
                    mp.solutions.drawing_utils.draw_landmarks(
                        img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 122, 255), thickness=1)
                    )
                    
                    # Normalize Data
                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x); y_.append(lm.y)
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))

                # Padding if data is insufficient (assuming model expects 84 features)
                if len(data_aux) < 84: 
                    data_aux.extend([0.0] * (84 - len(data_aux)))

                if model:
                    try:
                        proba = model.predict_proba([np.asarray(data_aux[:84])])
                        conf = np.max(proba)

                        if conf > 0.85:
                            raw_idx = int(np.argmax(proba))
                            # Handle prediction mapping
                            prediction = model.classes_[raw_idx]
                            char = self.labels_list.get(int(prediction), "?")

                            # --- INTERACTIVE LOGIC ---
                            if char == self.prev: 
                                self.frames += 1
                            else: 
                                self.frames = 0; self.prev = char

                            # 12 frames lock-in time
                            if self.frames == 12: 
                                if char != self.last_char:
                                    self.generated_text += char + " "
                                    self.last_char = char

                            # Glow Effect Label on Video (Letter near hand)
                            txt_x, txt_y = int(min(x_) * w_img), int(min(y_) * h_img) - 20
                            cv2.putText(img, char, (txt_x, txt_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 122, 255), 2)
                    except Exception as e:
                        print(f"Prediction Error: {e}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MAIN INTERFACE ---
st.markdown("<h3 style='text-align: center;'>Neural View Translation</h3>", unsafe_allow_html=True)
st.markdown("---")

# Layout: Left for Video (3/5 width), Right for Text Box (2/5 width)
col_video, col_results = st.columns([3, 2]) 

with col_video:
    # Video Streamer
    ctx = webrtc_streamer(
        key="neural-view",
        video_processor_factory=ModernProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

with col_results:
    st.markdown("### 📝 Translated Text")
    
    # Placeholder for the text box (This will be updated by the loop below)
    output_box = st.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Clear Button
    if st.button("✨ Clear Text", use_container_width=True):
        if ctx.video_processor:
            ctx.video_processor.generated_text = ""
            ctx.video_processor.last_char = ""
            st.toast("Translation cleared!", icon="🗑️")

# --- REAL-TIME UPDATE LOOP ---
# This continuously updates the side text box while video is playing
if ctx.state.playing:
    while True:
        if ctx.video_processor:
            current_text = ctx.video_processor.generated_text
            
            # Update the side box with styled HTML
            output_box.markdown(
                f"""
                <div style='
                    background-color: #f0f2f6; 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 5px solid #007aff;
                    font-family: sans-serif;
                    font-size: 20px; 
                    min-height: 200px;
                    color: #333;'>
                    {current_text if current_text else "<i style='color:gray;'>Waiting for gesture...</i>"}
                </div>
                """, 
                unsafe_allow_html=True
            )
        time.sleep(0.1)