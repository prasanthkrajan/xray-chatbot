import streamlit as st
from PIL import Image
import torch
import torchxrayvision as xrv
import numpy as np
import io
from torchvision import transforms

st.set_page_config(page_title="X-ray Chatbot PoC", layout="centered")

# Initialize the model (do this once at startup)
@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model

# Initialize the model
model = load_model()

# Define image preprocessing
def preprocess_image(image):
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 224x224
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    
    # Add channel dimension
    img_array = img_array[..., np.newaxis]
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

st.title("🩻 X-ray Diagnostic Chatbot (Proof of Concept)")
st.markdown("Upload a chest X-ray image (from camera or file), and chat with the AI assistant.")

image_file = st.file_uploader("Upload your chest X-ray image", type=['jpg', 'jpeg', 'png'])

if image_file:
    # Load and preprocess the image
    image = Image.open(image_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    # Preprocess the image
    img_tensor = preprocess_image(image)

    # Get predictions
    with torch.no_grad():
        outputs = torch.sigmoid(model(img_tensor))
        outputs = outputs[0].numpy()

    # Get condition names and their predictions
    conditions = xrv.datasets.default_pathologies
    findings = {condition: float(score) for condition, score in zip(conditions, outputs)}
    
    # Filter out conditions with very low confidence
    findings = {k: v for k, v in findings.items() if v > 0.1}

    st.subheader("🧪 AI Findings:")
    for condition, confidence in findings.items():
        st.write(f"**{condition}** – Confidence: {int(confidence * 100)}%")

    # Simulate chat interaction
    st.markdown("### 🤖 Chatbot")
    st.write("Would you like me to explain what these findings mean?")

    # Dynamic buttons based on findings
    for condition in findings.keys():
        if st.button(f"Explain {condition.lower()}"):
            st.info(f"Information about {condition}: This is a placeholder. In a real implementation, we would provide detailed medical information about this condition.")

    if st.button("Show symptoms to watch for"):
        st.warning("Common symptoms include:\n- Cough\n- Fever or chills\n- Chest pain\n- Difficulty breathing\n- Fatigue")

    if st.button("What should I do next?"):
        st.success("This tool is for educational purposes only. If you or someone has these symptoms, it's best to consult a medical professional for further advice.")

else:
    st.info("Please upload a chest X-ray image to begin.")
