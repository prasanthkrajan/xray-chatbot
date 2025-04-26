import streamlit as st
from PIL import Image
import torch
import torchxrayvision as xrv
import numpy as np
import io
from torchvision import transforms
import logging
import skimage.transform

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
        # Load the pre-trained model
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
        
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
            logger.info("Converted to grayscale")
        
        # Convert to numpy array and normalize to [0, 255] range
        img_array = np.array(image)
        img_array = img_array.astype(float)
        
        # Normalize to [-1024, 1024] range as per torchxrayvision
        img_array = xrv.datasets.normalize(img_array, 255)
        
        # Resize to 224x224 (model's expected input size)
        img_array = skimage.transform.resize(img_array, (224, 224))
        
        # Add channel dimension
        img_array = img_array[None, ...]
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).float()
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        logger.info(f"Final tensor shape: {img_tensor.shape}")
        logger.info(f"Tensor stats - Mean: {torch.mean(img_tensor):.2f}, Std: {torch.std(img_tensor):.2f}")
        
        return img_tensor
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def get_condition_description(condition):
    """
    Get a detailed description for each condition the model can detect.
    """
    descriptions = {
        'Atelectasis': 'A complete or partial collapse of the entire lung or area (lobe) of the lung. It occurs when the tiny air sacs (alveoli) within the lung become deflated or possibly filled with fluid.',
        'Cardiomegaly': 'Enlargement of the heart, which can be caused by various conditions including high blood pressure, heart valve disease, or heart failure.',
        'Consolidation': 'A region of normally compressible lung tissue that has filled with liquid instead of air. It may appear as a dense area on chest X-rays.',
        'Edema': 'Buildup of fluid in the lungs\' air sacs, which can make breathing difficult. Often caused by heart problems.',
        'Effusion': 'A condition in which excess fluid builds up in the pleural space, the space between the lungs and the chest wall.',
        'Emphysema': 'A lung condition that causes shortness of breath. In emphysema, the air sacs in the lungs (alveoli) are damaged.',
        'Fibrosis': 'Scarring of lung tissue, which can make breathing increasingly difficult. Various factors can lead to pulmonary fibrosis.',
        'Hernia': 'A condition in which an organ pushes through an opening in the muscle or tissue that holds it in place.',
        'Infiltration': 'An abnormal substance that has accumulated in the lung tissue, such as pus, blood, or water.',
        'Mass': 'An abnormal growth or tumor that may require further investigation to determine if it\'s benign or malignant.',
        'Nodule': 'A small round or oval-shaped growth in the lung. While many are benign, some may need further evaluation.',
        'Pleural_Thickening': 'A condition where the pleura (the lining that covers the lungs) becomes thicker, often due to asbestos exposure or other factors.',
        'Pneumonia': 'An infection that inflames the air sacs in one or both lungs, which may fill with fluid.',
        'Pneumothorax': 'A collapsed lung occurs when air leaks into the space between the lung and chest wall.',
        'Support Devices': 'Medical devices visible in the X-ray, such as pacemakers, tubes, or other supportive equipment.',
    }
    return descriptions.get(condition, 'No detailed description available for this condition.')

def get_predictions(img_tensor):
    try:
        logger.info("Starting prediction")
        
        # Ensure model is in evaluation mode
        model.eval()
        
        with torch.no_grad():
            # Get raw predictions
            outputs = model(img_tensor)
            
            # Convert to probabilities using sigmoid
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
            # Get predictions for each pathology
            predictions = {}
            for i, pathology in enumerate(model.pathologies):
                predictions[pathology] = float(probabilities[i])
                logger.info(f"{pathology}: {predictions[pathology]:.3f}")
            
            # Sort predictions by probability
            sorted_predictions = dict(sorted(predictions.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True))
            
            return sorted_predictions
            
    except Exception as e:
        logger.error(f"Error in get_predictions: {str(e)}")
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
        predictions = get_predictions(img_tensor)

        # Get condition names and their predictions
        conditions = xrv.datasets.default_pathologies
        findings = {condition: float(score) for condition, score in predictions.items()}
        
        # Log raw predictions for debugging
        logger.info("Raw predictions:")
        for condition, score in findings.items():
            logger.info(f"{condition}: {score:.3f}")
        
        # Filter and sort findings by confidence with a higher threshold
        # Use dynamic thresholding based on the distribution of predictions
        prediction_mean = np.mean(list(findings.values()))
        prediction_std = np.std(list(findings.values()))
        
        # Set threshold as mean + 0.5 * std for more selective results
        base_threshold = prediction_mean + 0.5 * prediction_std
        logger.info(f"Dynamic threshold calculated: {base_threshold:.3f}")
        
        def get_threshold(condition):
            high_threshold_conditions = {'Mass', 'Pneumonia', 'Cardiomegaly'}
            # Add 0.1 to threshold for serious conditions
            return base_threshold + 0.1 if condition in high_threshold_conditions else base_threshold
        
        significant_findings = {
            k: v for k, v in findings.items() 
            if v > get_threshold(k)
        }
        sorted_findings = dict(sorted(significant_findings.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Found {len(sorted_findings)} significant findings after thresholding")

        if len(sorted_findings) > 0:
            st.subheader("🔍 AI Analysis Results:")
            
            # Display findings with confidence bars and color coding
            for condition, confidence in sorted_findings.items():
                confidence_percentage = int(confidence * 100)
                
                # Color code based on confidence level
                if confidence > 0.7:
                    color = "🔴 High Confidence"
                elif confidence > 0.5:
                    color = "🟡 Medium Confidence"
                else:
                    color = "⚪ Low Confidence"
                
                st.markdown(f"### {color} - {condition}")
                st.progress(confidence)
                st.markdown(f"**Confidence: {confidence_percentage}%**")
                
                # Show condition description
                with st.expander("Learn more about this condition"):
                    st.markdown(get_condition_description(condition))
                    st.markdown("""
                    **Note:** The confidence score indicates the AI model's certainty in detecting this condition. 
                    Higher scores suggest stronger evidence in the X-ray image, but all findings should be 
                    verified by a qualified healthcare professional.
                    """)
                st.markdown("---")

            # Overall summary
            st.subheader("📋 Summary")
            high_confidence_findings = [k for k, v in sorted_findings.items() if v > 0.6]
            medium_confidence_findings = [k for k, v in sorted_findings.items() if 0.4 <= v <= 0.6]
            
            if high_confidence_findings:
                st.markdown("**🔴 Key Findings (High Confidence):**")
                for finding in high_confidence_findings:
                    st.markdown(f"- {finding}")
            
            if medium_confidence_findings:
                st.markdown("**🟡 Additional Findings (Medium Confidence):**")
                for finding in medium_confidence_findings:
                    st.markdown(f"- {finding}")
            
            st.warning("""
            ⚠️ **Important Medical Notice:**
            1. This AI analysis is for educational and demonstration purposes only
            2. The results should NOT be used for medical diagnosis
            3. The model may not detect all conditions present
            4. False positives and false negatives are possible
            5. Always consult with qualified healthcare professionals
            6. Further medical evaluation is needed for proper diagnosis
            """)
            
        else:
            st.info("""
            No significant findings detected with high confidence. However, please note that:
            - This does not guarantee the absence of medical conditions
            - The AI model has limitations and may miss subtle findings
            - Always consult healthcare professionals for proper medical evaluation
            """)
            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        st.error(f"An error occurred while processing the image: {str(e)}")
else:
    st.info("Please upload a chest X-ray image to begin.")
