# %%
import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np
import threading

# %%
model = YOLO('best (5).pt')

class_names = ['Buffalo', 'Elephant', 'Rhinoceros', 'zebra']

st.title("Animal")

# %%
def detect_traffic_signs(frame):
    results = model(frame, conf=0.75)
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


# %%
def video_capture():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_traffic_signs(frame)
        cv2.imshow('Traffic Sign Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

camera_active = False

# %%
camera_active = False
def start_camera():
    global camera_active
    camera_active = True
    video_thread = threading.Thread(target=video_capture)
    video_thread.start()

# Streamlit UI
if st.button("Start Camera"):
    start_camera()

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png'], key='file_uploader')

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    image_np = np.array(image)

    crop_button = st.button('Crop Image')

    if crop_button:
        def get_mouse_coords(event, x, y, flags, param):
            global coords, cropping
            if event == cv2.EVENT_LBUTTONDOWN:
                coords = [(x, y)]
                cropping = True
            elif event == cv2.EVENT_LBUTTONUP:
                coords.append((x, y))
                cropping = False
                cv2.rectangle(image_np, coords[0], coords[1], (0, 255, 0), 2)
                cv2.imshow("Image", image_np)

        coords = []
        cropping = False

        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", get_mouse_coords)
        
        while True:
            cv2.imshow("Image", image_np)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cv2.destroyAllWindows()

        if len(coords) == 2:
            x1, y1 = coords[0]
            x2, y2 = coords[1]

            cropped_image = image_np[y1:y2, x1:x2]
            st.image(cropped_image, caption='Cropped Image', use_column_width=True)

            zoom = st.slider('Zoom', 1, 10, 1)
            new_height, new_width = cropped_image.shape[0] * zoom, cropped_image.shape[1] * zoom
            zoomed_image = cv2.resize(cropped_image, (new_width, new_height))

            results = model(zoomed_image, conf=0.5)

            predictions = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0]
                    cls = box.cls[0]

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    label = f'{class_names[int(cls)]}: {conf:.2f}'
                    predictions.append(label)

                    cv2.rectangle(zoomed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(zoomed_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            st.image(zoomed_image, caption='Processed Image', use_column_width=True)
            st.write("Predictions:")
            for prediction in predictions:
                st.write(prediction)


