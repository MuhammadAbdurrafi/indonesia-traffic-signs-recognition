import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load YOLOv8n model (trained, in best.pt)
@st.cache_resource
def load_model():
    return YOLO('best.pt')  # make sure 'best.pt' is in the same folder

model = load_model()

st.title("🔍 YOLOv8 Traffic Sign Recognition (CPU - Streamlit App)")
st.write("Upload a image below  and see what traffic sign it is using our custom model.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Optional resize to 416x416 for speed (if not already)
    image_resized = image.resize((416, 416))
    
    st.image(image_resized, caption="Uploaded Image (416x416)", use_column_width=True)

    with st.spinner("Running detection..."):
        # img_np = np.array(image_resized)
        results = model(image_resized, device="cpu")
        # force CPU
        annotated_img = results[0].plot()  # get numpy array with bounding boxes

        st.image(annotated_img, caption="Detected Objects", use_column_width=True)

        # Optional: show labels and confidences
        st.subheader("📋 Detected Classes")
        names = model.names
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            st.write(f"• {names[cls_id]} ({conf:.2%})")
