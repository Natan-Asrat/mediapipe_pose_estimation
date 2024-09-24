import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
from pose_detector.module import PoseEstimator

class CustomVideoProcessor(VideoTransformerBase):
    def __init__(self, detector):
        self.detector = detector

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format
        drawn = self.detector.findPoses(img)
        return av.VideoFrame.from_ndarray(drawn, format="bgr24")

# Main application logic
st.title("MediaPipe Pose Detection Live Stream")
st.text("Live Pose Detection  by Natan Asrat.")

@st.cache_resource(show_spinner=True)
def load_model():
    st.text("Loading MediaPipe Detector...")
    detector = PoseEstimator()
    return detector

detector = load_model()



# Use WebRTC for live video stream
webrtc_streamer(key="example",  video_processor_factory=lambda: CustomVideoProcessor(detector))
