import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import time

# Load the model
# model = YOLO('best (5).pt') #precision 97%
# model = YOLO('best (5).pt') #precision 95%
class_names = ['Buffalo', 'Elephant', 'Rhinoceros', 'Zebra']

st.title("Animal Detection")

css = f'''
    <style>
        .stApp > header {{
            background-color: transparent;
        }}

    .stApp {{
        margin: auto;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        overflow: auto;
        background: linear-gradient(315deg, #FFB862 15%, #E8816D 35%, #B15B75 55%, #6C4369 75%, #2c2c48 95%);
        animation: gradient 10s ease infinite;
        background-size: 400% 400%;
        background-attachment: fixed;
    }}

    img {{
        width: 100px;
    }}

    @keyframes gradient {{
        0% {{
            background-position: 0% 50%;
        }}
        50% {{
            background-position: 100% 50%;
        }}
        100% {{
            background-position: 0% 50%;
        }}
    }}

    </style>
    '''
st.markdown(css, unsafe_allow_html=True)

# Function to detect animals
def detect_animals(frame):
    results = model(frame, conf=0.5)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0]
            cls = box.cls[0]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f'{class_names[int(cls)]}: {conf:.2f}'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Streamlit UI
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

col1, col2, col3, col4, col5= st.columns(5)

with col1:
    st.image("upload_logo.png")

with col3:
   camera = st.toggle("")

with col5:
   st.image("camera.png")

if camera != True:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

cap = None

# Function to start the camera stream
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Cannot open camera")
        return

    frame_width, frame_height = 640, 480  # Define the desired frame size
    frame_placeholder = st.empty()  # Placeholder for updating the frame

    while st.session_state.camera_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        frame = detect_animals(frame)
        frame_resized = cv2.resize(frame, (frame_width, frame_height))  # Resize frame
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        # Update the image in the Streamlit app
        frame_placeholder.image(image, channels="RGB", use_column_width=True)
        time.sleep(0.1)  # Adjust the sleep time if necessary

    cap.release()

if camera:
    st.session_state.camera_active = True
    st.write("Camera Active: True")
    start_camera()

if camera != True:
    st.session_state.camera_active = False
    st.write("Camera is off. Click 'Start Camera' to start the camera.")
    if cap is not None:
        cap.release()

# Process uploaded image
if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    uploaded_image = np.array(uploaded_image)
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)

    # Detect animals in the uploaded image
    detected_image = detect_animals(uploaded_image)

    # Convert back to RGB for displaying
    detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    detected_image_pil = Image.fromarray(detected_image)

    st.image(detected_image_pil, caption='Uploaded Image with Detections', use_column_width=True)

frame_placeholder = st.empty()

# Correct path for the background image



st.markdown(css, unsafe_allow_html=True)
