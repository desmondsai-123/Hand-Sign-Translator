Desmond, [14/1/2026 11:51 PM]
import pickle
import cv2
import mediapipe as mp
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="VisionAI - Sign Translator",
    page_icon="✨",
    layout="centered",
)

# --- ADVANCED AESTHETIC CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;600&family=Inter:wght@300;500&display=swap');

    /* BACKGROUND GRADIENT */
    .stApp {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        font-family: 'SF Pro Display', 'Inter', sans-serif;
    }

    /* GLASS CARD EFFECT */
    div[data-testid="stVerticalBlock"] > div:has(div.stVideo) {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(15px);
        border-radius: 35px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.1);
    }

    /* GLOWING TITLE */
    h1 {
        background: linear-gradient(to right, #2c3e50, #007AFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        font-size: 3rem !important;
        letter-spacing: -1.5px;
    }

    /* INTERACTIVE STATUS BADGE */
    .status-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-bottom: 20px;
    }
    .badge {
        padding: 8px 16px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-blue { background: #007AFF; color: white; box-shadow: 0 4px 15px rgba(0, 122, 255, 0.3); }
    .badge-green { background: #34C759; color: white; box-shadow: 0 4px 15px rgba(52, 199, 89, 0.3); }

    /* STREAMLIT BUTTONS */
    .stButton>button {
        background: rgba(255, 255, 255, 0.8);
        color: #007AFF;
        border: 1px solid #007AFF;
        border-radius: 15px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
    }
    .stButton>button:hover {
        background: #007AFF;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 122, 255, 0.2);
    }

    /* HIDE DEFAULTS */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- APP UI START ---
st.markdown("<h1>VisionAI</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="status-container">
    <span class="badge badge-blue">Neural Engine Active</span>
    <span class="badge badge-green">v2.0 Interface</span>
</div>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    try:
        with open('./model.p', 'rb') as f:
            model_dict = pickle.load(f)
        return model_dict['model']
    except:
        st.error("Model brain (model.p) missing. Please train first!")
        return None

model = load_model()

# --- INTERACTIVE PROCESSING CLASS ---
class ModernProcessor(VideoTransformerBase):
    def init(self):
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.labels_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.generated_text = ""
        self.prev = None
        self.frames = 0
        self.last_char = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h_img, w_img, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        # Draw Blurred Message Overlay (Glassmorphism look)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h_img-100), (w_img, h_img), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)

Desmond, [14/1/2026 11:51 PM]
if results.multi_hand_landmarks:
            data_aux, x_, y_ = [], [], []
            for hand_landmarks in results.multi_hand_landmarks:
                # Soft White UI for Landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 122, 255), thickness=1)
                )
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x); y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_)); data_aux.append(lm.y - min(y_))

            if len(data_aux) < 84: data_aux.extend([0.0] * (84 - len(data_aux)))

            if model:
                proba = model.predict_proba([np.asarray(data_aux[:84])])
                conf = np.max(proba)
                
                if conf > 0.85:
                    raw_idx = int(np.argmax(proba))
                    char = self.labels_list[int(model.classes_[raw_idx])]

                    # INTERACTIVE LOGIC
                    if char == self.prev: self.frames += 1
                    else: self.frames = 0; self.prev = char

                    if self.frames == 12: # Locked in
                        if char != self.last_char:
                            self.generated_text += char + " "
                            self.last_char = char

                    # Glow Effect Label
                    txt_x, txt_y = int(min(x_) * w_img), int(min(y_) * h_img) - 20
                    cv2.putText(img, char, (txt_x, txt_y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 122, 255), 2)

        # Modern Text Output
        display_text = "Ready to translate..." if not self.generated_text else self.generated_text
        cv2.putText(img, display_text, (40, h_img-40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (40, 40, 40), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- MAIN INTERFACE ---
ctx = webrtc_streamer(
    key="neural-view",
    video_processor_factory=ModernProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("<br>", unsafe_allow_html=True)

# INTERACTIVE CONTROLS
c1, c2, c3 = st.columns([1,1,1])
with c2:
    if st.button("✨ Reset Session"):
        if ctx.video_processor:
            ctx.video_processor.generated_text = ""
            st.toast("Message cleared!", icon='✅')

# FOOTER CAPTION
st.markdown("""
<div style='text-align: center; color: #86868b; font-size: 0.8rem; margin-top: 50px;'>
    Developed for Accessibility • Powered by Computer Vision
</div>
""", unsafe_allow_html=True)
