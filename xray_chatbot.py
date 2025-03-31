import streamlit as st
from PIL import Image
import random

st.set_page_config(page_title="X-ray Chatbot PoC", layout="centered")

st.title("🩻 X-ray Diagnostic Chatbot (Proof of Concept)")
st.markdown("Upload a chest X-ray image (from camera or file), and chat with the AI assistant.")

image_file = st.file_uploader("Upload your chest X-ray image", type=['jpg', 'jpeg', 'png'])

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Generate random findings
    possible_conditions = [
        "Pneumonia",
        "Pleural Effusion",
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Atelectasis",
        "Pneumothorax"
    ]
    
    # Randomly select 2-4 conditions and assign random confidence scores
    num_conditions = random.randint(2, 4)
    selected_conditions = random.sample(possible_conditions, num_conditions)
    findings = {condition: round(random.uniform(0.3, 0.95), 2) for condition in selected_conditions}

    st.subheader("🧪 AI Findings:")
    for condition, confidence in findings.items():
        st.write(f"**{condition}** – Confidence: {int(confidence * 100)}%")

    # Simulate chat interaction
    st.markdown("### 🤖 Chatbot")
    st.write("Would you like me to explain what these findings mean?")

    if st.button("Yes, explain pneumonia"):
        st.info("Pneumonia is an infection that causes inflammation in the air sacs of the lungs. These air sacs may fill with fluid or pus, leading to cough, fever, and difficulty breathing.")

    if st.button("What is pleural effusion?"):
        st.info("Pleural effusion is when fluid builds up around the lungs. It can cause chest pain and trouble breathing. It's important to see a doctor to find the cause.")

    if st.button("Show symptoms to watch for"):
        st.warning("Common symptoms include:\n- Cough\n- Fever or chills\n- Chest pain\n- Difficulty breathing\n- Fatigue")

    if st.button("What should I do next?"):
        st.success("This tool is for educational purposes only. If you or someone has these symptoms, it's best to consult a medical professional for further advice.")

else:
    st.info("Please upload a chest X-ray image to begin.")
