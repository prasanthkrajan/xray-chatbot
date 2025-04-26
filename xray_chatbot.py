import streamlit as st
from PIL import Image
import torch
import torchxrayvision as xrv
import numpy as np
import io
from torchvision import transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="X-ray Diagnostic Chatbot", layout="centered")

def is_xray_image(image):
    """
    Validate if the uploaded image is an X-ray based on certain characteristics
    """
    try:
        # Convert to grayscale if not already
        if image.mode != 'L':
            logger.info(f"Converting image from {image.mode} to grayscale")
            image = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        logger.info(f"Image array shape: {img_array.shape}")
        
        # Calculate basic statistics
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        logger.info(f"Image statistics - Mean: {mean_intensity:.2f}, Std: {std_intensity:.2f}")
        
        # X-rays typically have:
        # - Mean intensity between 30-220 (out of 255) - widened range
        # - Standard deviation > 15 (indicating reasonable contrast)
        if mean_intensity < 30 or mean_intensity > 220:
            logger.info("Image rejected: Mean intensity out of range")
            return False
        if std_intensity < 15:
            logger.info("Image rejected: Standard deviation too low")
            return False
        
        # Check if image has reasonable contrast distribution
        # Instead of just dark pixels, we'll check for a good distribution
        hist, _ = np.histogram(img_array, bins=64, range=(0, 256))
        dark_region = np.sum(hist[:16])  # First quarter of histogram
        light_region = np.sum(hist[-16:])  # Last quarter of histogram
        
        # Ensure there's some balance between dark and light regions
        dark_ratio = dark_region / img_array.size
        light_ratio = light_region / img_array.size
        logger.info(f"Dark ratio: {dark_ratio:.2f}, Light ratio: {light_ratio:.2f}")
        
        if dark_ratio < 0.05 or light_ratio < 0.05:
            logger.info("Image rejected: Poor contrast distribution")
            return False
        
        logger.info("Image validated as X-ray")
        return True
    except Exception as e:
        logger.error(f"Error in is_xray_image: {str(e)}")
        return False

# Initialize the model (do this once at startup)
@st.cache_resource
def load_model():
    try:
        logger.info("Loading model...")
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Initialize the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Define image preprocessing
def preprocess_image(image):
    try:
        logger.info("Starting image preprocessing")
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
            logger.info("Converted to grayscale")
        
        # Resize to 224x224
        original_size = image.size
        image = image.resize((224, 224))
        logger.info(f"Resized image from {original_size} to (224, 224)")
        
        # Convert to numpy array and normalize
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        logger.info(f"Normalized array shape: {img_array.shape}")
        
        # Add channel dimension
        img_array = img_array[..., np.newaxis]
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0)
        logger.info(f"Final tensor shape: {img_tensor.shape}")
        
        return img_tensor
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

st.title("🩻 X-ray Diagnostic Chatbot")
st.markdown("Upload a chest X-ray image (from camera or file), and chat with the AI assistant.")

image_file = st.file_uploader("Upload your chest X-ray image", type=['jpg', 'jpeg', 'png'])

if image_file:
    try:
        logger.info(f"Processing uploaded file: {image_file.name}")
        # Load and preprocess the image
        image = Image.open(image_file)
        logger.info(f"Opened image: size={image.size}, mode={image.mode}")
        
        # Validate if it's an X-ray
        if not is_xray_image(image):
            st.error("❌ This doesn't appear to be an X-ray image. Please upload a valid chest X-ray image.")
            st.stop()
        
        st.image(image, caption="Uploaded X-ray", use_column_width=True)
        
        # Preprocess the image
        img_tensor = preprocess_image(image)

        # Get predictions
        with torch.no_grad():
            outputs = torch.sigmoid(model(img_tensor))
            outputs = outputs[0].numpy()
            logger.info("Model prediction completed successfully")

        # Get condition names and their predictions
        conditions = xrv.datasets.default_pathologies
        findings = {condition: float(score) for condition, score in zip(conditions, outputs)}
        
        # Filter out conditions with very low confidence
        findings = {k: v for k, v in findings.items() if v > 0.1}
        logger.info(f"Found {len(findings)} significant findings")

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

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        st.error(f"An error occurred while processing the image: {str(e)}")
else:
    st.info("Please upload a chest X-ray image to begin.")
