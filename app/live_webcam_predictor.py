from pathlib import Path
from collections import deque
import threading
import queue
import time

try:
    import av
except ImportError:
    av = None

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import mediapipe as mp

try:
    from streamlit_webrtc import (
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )
except ImportError:
    webrtc_streamer = None
    VideoProcessorBase = object
    WebRtcMode = None


# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

IMG_SIZE = (128, 128)
CLASS_NAMES = ["Rock", "Paper", "Scissors"]
CLASS_ICONS = {
    "Rock": "✊",
    "Paper": "✋",
    "Scissors": "✌️",
}

# Portable model path resolution based on project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "rps_cnn.keras"


# ============================================================
# MODEL LOADING (CACHED ONCE)
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained Keras CNN model once and cache it."""
    if not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as exc:
        print(f"[Model Load Error]: {exc}")
        return None


# ============================================================
# LIVE WEBRTC VIDEO PROCESSOR (MEDIAPIPE + CNN DECOUPLED)
# ============================================================

class RPSVideoProcessor(VideoProcessorBase):
    """
    WebRTC video processor that decouples video streaming from inference.
    Webcam frames in recv() are immediately queued without blocking.
    A background daemon thread handles MediaPipe hand detection,
    bounding box cropping with padding, and CNN inference.
    """

    latest_instance = None

    def __init__(self):
        RPSVideoProcessor.latest_instance = self
        self.lock = threading.Lock()
        self.model = load_model()

        # Shared prediction state
        self.prediction = "NO HAND DETECTED"
        self.confidence = 0.0
        self.rock_probability = 0.0
        self.paper_probability = 0.0
        self.scissors_probability = 0.0
        self.probabilities = {"Rock": 0.0, "Paper": 0.0, "Scissors": 0.0}
        self.hand_detected = False
        self.latest_frame_timestamp = time.time()

        self.history = deque(maxlen=5)
        self.has_logged_error = False

        # Drop-oldest queue of maxsize=1 so worker always processes the newest frame
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True

        # Dedicated background inference worker thread
        self.worker_thread = threading.Thread(
            target=self.inference_worker,
            daemon=True,
        )
        self.worker_thread.start()

    def _set_no_hand(self):
        """Set state to NO HAND DETECTED and clear smoothing history."""
        self.history.clear()
        with self.lock:
            self.prediction = "NO HAND DETECTED"
            self.confidence = 0.0
            self.rock_probability = 0.0
            self.paper_probability = 0.0
            self.scissors_probability = 0.0
            self.probabilities = {"Rock": 0.0, "Paper": 0.0, "Scissors": 0.0}
            self.hand_detected = False
            self.latest_frame_timestamp = time.time()

    def inference_worker(self):
        """
        Background worker thread:
        Runs MediaPipe hand detection and crops hand region,
        then feeds cropped hand to CNN inference.
        """
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        last_inference_time = 0.0
        min_interval = 0.15  # Approx 6-7 FPS inference rate

        while self.running:
            try:
                # Wait for newest frame
                image = self.frame_queue.get(timeout=0.2)
                while True:
                    try:
                        image = self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            active_model = self.model if self.model is not None else load_model()
            if active_model is None:
                continue

            # Throttle inference rate to preserve CPU headroom
            now = time.monotonic()
            elapsed = now - last_inference_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            try:
                h, w, _ = image.shape

                # 1. BGR -> RGB using numpy slicing (no cv2)
                rgb = image[:, :, ::-1].copy()

                # 2. MediaPipe Hand Detection
                mp_results = hands.process(rgb)

                if mp_results.multi_hand_landmarks:
                    # Hand detected: compute bounding box
                    hand_landmarks = mp_results.multi_hand_landmarks[0]
                    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                    y_coords = [lm.y * h for lm in hand_landmarks.landmark]

                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)

                    box_w = max_x - min_x
                    box_h = max_y - min_y

                    # Add 25% margin / padding around the hand bounding box
                    pad_x = box_w * 0.25
                    pad_y = box_h * 0.25

                    # Clamp coordinates to frame boundaries
                    x1 = max(0, int(min_x - pad_x))
                    y1 = max(0, int(min_y - pad_y))
                    x2 = min(w, int(max_x + pad_x))
                    y2 = min(h, int(max_y + pad_y))

                    hand_crop = rgb[y1:y2, x1:x2]

                    if hand_crop.shape[0] > 10 and hand_crop.shape[1] > 10:
                        # Resize crop to 128x128 using PIL bilinear resampling
                        crop_pil = Image.fromarray(hand_crop).resize(
                            IMG_SIZE, Image.Resampling.BILINEAR
                        )
                        # Normalize to float32 [0, 1] and add batch dimension
                        tensor = np.expand_dims(
                            np.asarray(crop_pil, dtype=np.float32) / 255.0,
                            axis=0,
                        )

                        # 3. CNN inference
                        raw_probs = active_model(tensor, training=False).numpy()[0]
                        raw_probs = np.asarray(raw_probs, dtype=np.float32)

                        # 4. Temporal smoothing (deque maxlen=3)
                        self.history.append(raw_probs)
                        smoothed = np.mean(np.asarray(self.history), axis=0)

                        predicted_index = int(np.argmax(smoothed))
                        predicted_label = CLASS_NAMES[predicted_index]
                        confidence = float(smoothed[predicted_index] * 100.0)

                        r_prob = float(smoothed[0] * 100.0)
                        p_prob = float(smoothed[1] * 100.0)
                        s_prob = float(smoothed[2] * 100.0)

                        prob_dict = {
                            "Rock": r_prob,
                            "Paper": p_prob,
                            "Scissors": s_prob,
                        }

                        with self.lock:
                            self.prediction = predicted_label
                            self.confidence = confidence
                            self.rock_probability = r_prob
                            self.paper_probability = p_prob
                            self.scissors_probability = s_prob
                            self.probabilities = prob_dict
                            self.hand_detected = True
                            self.latest_frame_timestamp = time.time()
                    else:
                        self._set_no_hand()
                else:
                    # No hand detected: clear history and do NOT run CNN
                    self._set_no_hand()

                last_inference_time = time.monotonic()

            except Exception as exc:
                if not self.has_logged_error:
                    print(f"[RPS Inference Error]: {exc}")
                    self.has_logged_error = True

        hands.close()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Lightweight WebRTC video callback.
        Immediately queues the latest frame and returns without blocking.
        """
        image = frame.to_ndarray(format="bgr24")

        # Put newest frame into queue; discard unconsumed frame if full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self.frame_queue.put_nowait(image)
        except queue.Full:
            pass

        return frame

    def __del__(self):
        self.running = False


# ============================================================
# STREAMLIT UI MAIN FUNCTION
# ============================================================

def main():
    st.set_page_config(
        page_title="Rock Paper Scissors AI",
        page_icon="✊",
        layout="wide",
    )

    # Header and subtitle
    st.html(
        """<div style="text-align: center; padding: 8px 0 16px 0;">
            <h1 style="font-size: 2.6rem; font-weight: 800; margin: 0 0 6px 0; color: #f8fafc; letter-spacing: 1px;">
                ROCK &bull; PAPER &bull; SCISSORS
            </h1>
            <p style="font-size: 1.15rem; color: #94a3b8; margin: 0;">
                Real-time hand gesture classification using a custom CNN
            </p>
        </div>"""
    )

    st.divider()

    # Dependency check
    if av is None or webrtc_streamer is None:
        st.error("❌ Required WebRTC libraries (`streamlit-webrtc`, `av`) are not installed.")
        st.stop()

    model = load_model()

    # Status indicators
    status_left, status_right = st.columns(2)

    with status_left:
        if model is not None:
            st.success("🧠 **MODEL:** Loaded ✓")
        else:
            st.error("❌ **MODEL:** Model file not found")
            st.caption(f"Expected at: `{MODEL_PATH}`")

    with status_right:
        st.info("📷 **CAMERA:** Ready ✓")

    # Main two-column layout
    camera_column, prediction_column = st.columns([1.2, 1.0], gap="large")

    with camera_column:
        st.markdown("### 📷 LIVE CAMERA")
        st.caption(
            "Show your hand clearly to the camera. "
            "MediaPipe detects the hand boundary, and the CNN classifies the gesture in real-time."
        )

        webrtc_ctx = webrtc_streamer(
            key="rps-live-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=RPSVideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640, "max": 640},
                    "height": {"ideal": 480, "max": 480},
                    "frameRate": {"ideal": 15, "max": 20},
                },
                "audio": False,
            },
            async_processing=True,
        )

        if webrtc_ctx.state.playing:
            st.success("🟢 Live prediction is running")
        else:
            st.info("Click START to begin live prediction.")

    with prediction_column:
        st.markdown("### 🎯 PREDICTION")

        @st.fragment(run_every=0.25)
        def render_prediction_panel():
            is_playing = bool(webrtc_ctx and webrtc_ctx.state.playing)
            processor = None

            if is_playing:
                processor = webrtc_ctx.video_processor or RPSVideoProcessor.latest_instance

            if is_playing and processor is not None:
                with processor.lock:
                    pred = processor.prediction
                    conf = processor.confidence
                    r_prob = processor.rock_probability
                    p_prob = processor.paper_probability
                    s_prob = processor.scissors_probability
                    hand_found = processor.hand_detected

                if hand_found:
                    icon = CLASS_ICONS.get(pred, "")
                    with st.container(border=True):
                        st.caption("CNN PREDICTION")
                        st.markdown(f"## {icon} {pred.upper()}")
                        st.metric(label="Confidence", value=f"{conf:.1f}%")
                        st.progress(min(max(conf / 100.0, 0.0), 1.0))
                else:
                    with st.container(border=True):
                        st.caption("CNN PREDICTION")
                        st.markdown("## ✋ NO HAND DETECTED")
                        st.info("Show your hand clearly to the camera.")
                        st.metric(label="Confidence", value="0.0%")
                        st.progress(0.0)

                st.markdown("#### Class probabilities")

                # ✊ Rock row
                col_r1, col_r2 = st.columns([2, 1])
                with col_r1:
                    st.write("✊ **Rock**")
                with col_r2:
                    st.write(f"**{r_prob:.1f}%**")
                st.progress(min(max(r_prob / 100.0, 0.0), 1.0))

                # ✋ Paper row
                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    st.write("✋ **Paper**")
                with col_p2:
                    st.write(f"**{p_prob:.1f}%**")
                st.progress(min(max(p_prob / 100.0, 0.0), 1.0))

                # ✌️ Scissors row (ALWAYS visible)
                col_s1, col_s2 = st.columns([2, 1])
                with col_s1:
                    st.write("✌️ **Scissors**")
                with col_s2:
                    st.write(f"**{s_prob:.1f}%**")
                st.progress(min(max(s_prob / 100.0, 0.0), 1.0))

            else:
                with st.container(border=True):
                    st.markdown("### 📷 Camera Ready")
                    st.write("Click **START** on the camera view to begin live gesture prediction.")
                    st.caption("The custom CNN classifies cropped hand regions at 128×128 RGB resolution.")

                st.markdown("#### Class probabilities")
                for name in CLASS_NAMES:
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.write(f"{CLASS_ICONS[name]} **{name}**")
                    with c2:
                        st.write("**0.0%**")
                    st.progress(0.0)

        render_prediction_panel()

    # Model Information footer
    st.divider()
    st.markdown("### 🧠 MODEL INFORMATION")

    info1, info2, info3, info4 = st.columns(4)

    with info1:
        st.metric("Model", "RPS CNN")

    with info2:
        st.metric("Input", "128 × 128 × 3")

    with info3:
        st.metric("Classes", "Rock / Paper / Scissors")

    with info4:
        st.metric("Recorded test accuracy", "96.51%")

    st.caption(
        "Architecture: Custom CNN + Dropout. "
        "Input preprocessing: 128×128 RGB normalized to [0, 1]. "
        "Class mapping: 0 = Rock, 1 = Paper, 2 = Scissors. "
        "Hand localization: MediaPipe Hands (detector gate)."
    )


if __name__ == "__main__":
    main()