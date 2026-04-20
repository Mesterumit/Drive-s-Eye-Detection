import streamlit as st
import cv2
import torch
import time
from torchvision import transforms
from PIL import Image
# importing the empty model from model.py
from model import EyeCNN

# Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = EyeCNN().to(device)
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
model.eval()



transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Face & eye detectors
# Why: OpenCV tools to find face & eyes
#      in camera frame before model predicts

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
)

# Session state
# Why: Streamlit reruns entire script
#      on every interaction!
#      session_state remembers values
#      between reruns
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'closed_counter' not in st.session_state:
    st.session_state.closed_counter = 0

# UI
st.title("👁️ Eye Detection System")

# Buttons

col1, col2 = st.columns(2)
with col1:
    start = st.button("▶️ Start Camera")
with col2:
    stop = st.button("⏹️ Stop Camera")

if start:
    # Memory:
    # remembers values between reruns
    st.session_state.camera_on = True
    st.session_state.closed_counter = 0
if stop:
    st.session_state.camera_on = False
    st.session_state.closed_counter = 0

# Placeholders
frame_placeholder = st.empty()
status_placeholder = st.empty()
alert_placeholder = st.empty()


# Alert threshold
ALERT_THRESHOLD = 15

# Camera loop
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0) # opend the camera

    while st.session_state.camera_on:
        ret, frame = cap.read()  # geting the frame from camera and " ret is true or false(didi it work)", frame is image(face)
        if not ret:
            st.error("❌ Camera failed!")
            break

        # Grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray


        # Detect faces
        faces = face_cascade.detectMultiScale( # find face
            gray,
            scaleFactor=1.1, # to capture diffirente size of images(faces)
            minNeighbors=4,
            minSize=(100, 100) # ignores anything smaller than 100*100 pixels
        )

        label = 0
        eye_found = False

        # Loop through faces
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # crop the face from image
            face_crop = gray[y:y+h, x:x+w]  # crop face

            # Detect eyes inside face
            eyes = eye_cascade.detectMultiScale(
                face_crop,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )

            for (ex, ey, ew, eh) in eyes:
                # Draw eye rectangle
                cv2.rectangle(
                    frame,
                    (x+ex, y+ey),
                    (x+ex+ew, y+ey+eh),
                    (0, 255, 0), 2
                )

                # Crop eye for model
                eye_img = face_crop[ey:ey+eh, ex:ex+ew]
                
                eye_pil = Image.fromarray(eye_img) #it turns into python image for tensor
                tensor = transform(eye_pil).unsqueeze(0).to(device) # prepare for model

                # Predict
                with torch.no_grad():
                    output = model(tensor) # predict the if it sleep or awake
                    _, predicted = torch.max(output, 1)
                    label = predicted.item()

                eye_found = True
                break  # use first eye only
            break  # use first face only

       
        # eyes status chcek
        if not eye_found or label == 1:   # closed or not found
            st.session_state.closed_counter += 1
            status_placeholder.error("👁️ SLEEPY/CLOSED ❌")
        else:                              # open
            st.session_state.closed_counter = 0
            status_placeholder.success("👁️ AWAKE/OPEN ✅")

        # Alert after 15 second
        if st.session_state.closed_counter >= ALERT_THRESHOLD:
            alert_placeholder.warning("🚨 ALERT! Eyes closed too long!")
        else:
            alert_placeholder.empty()

        # Show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", width=700)

        time.sleep(0.03) # wait 0.03 seconds between frames

    cap.release() # close camera when the button is clicked. 