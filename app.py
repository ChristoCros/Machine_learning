import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import time  # Import pour le chronométrage

# Configuration de la page
st.set_page_config(page_title="Toubib AI - Détection des lésions de la peau", layout="wide")
st.markdown("""
    <style>
        .block-container {padding-top: 3rem; padding-bottom: 0rem;}
        .element-container {margin-bottom: -0.5rem;}
        stProgress > div > div > div > div { height: 10px; }
        p { margin-bottom: 0rem; font-size: 0.9rem; }
        .stAlertContainer.st-bb { padding-left: 0.5rem; }
        .stAlertContainer.st-ba { padding-bottom: 0.5rem; }
        .stAlertContainer.st-b9 { padding-right: 0.5rem; }
        .stAlertContainer.st-b8 { padding-top: 0.5rem; }
        .st-emotion-cache-1s8qyds { margin-bottom: 0; }
        .st-emotion-cache-1s8qyds hr { margin: .5em 0px;}
        .inference-text { color: #666; font-size: 0.8rem !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🩺 Toubib AI - Détection des lésions de la peau")

# --- CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_models():
    onnx_path = ".venv/toubib_AI_training.onnx"
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    model_pt = YOLO(".venv/toubib_AI_training.pt") 
    return session, model_pt

try:
    session, model_pt = load_models()
    classes = ['Keratose_actinique', 'Carcinome_basocellulaire', 'Keratose_benigne', 
               'Dermatofibrome', 'Melanome', 'Naevus', 'Lesion_vasculaire']
except Exception as e:
    st.error(f"Erreur de chargement des modèles : {e}")

# --- UI ---
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    main_col1, main_col2 = st.columns([2, 1]) 
    
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    
    # --- PRÉTRAITEMENT POUR ONNX ---
    img_resized = cv2.resize(img_array, (224, 224))
    img_final = img_resized.astype(np.float32) / 255.0
    img_final_onnx = img_final.transpose(2, 0, 1)
    img_final_onnx = np.expand_dims(img_final_onnx, axis=0)

    # --- INFERENCE ONNX (Chronométrée) ---
    start_onnx = time.perf_counter()
    
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_final_onnx})
    logits = output[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    top_idx = np.argmax(probs)
    
    end_onnx = time.perf_counter()
    onnx_time = (end_onnx - start_onnx) * 1000 # Temps en ms

    # --- PARTIE GAUCHE : VISUELS ---
    with main_col1:
        st.subheader("🔍 Analyse visuelle & Focus IA")
        sub_col1, sub_col2 = st.columns(2)
        
        with sub_col1:
            st.image(image, use_container_width=True, caption="Photo originale")

        with sub_col2:
            # Génération automatique du Grad-CAM (Chronométrée)
            with st.spinner("Calcul Grad-CAM..."):
                start_grad = time.perf_counter()
                
                model_pt.model.train() 
                for param in model_pt.model.parameters():
                    param.requires_grad = True
                
                target_layers = [model_pt.model.model[-2]]
                input_tensor = torch.from_numpy(img_final_onnx).float()
                input_tensor.requires_grad = True 
                
                try:
                    cam = GradCAM(model=model_pt.model, target_layers=target_layers)
                    targets = [ClassifierOutputTarget(top_idx)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                    cam_image = show_cam_on_image(img_resized / 255.0, grayscale_cam, use_rgb=True)
                    
                    end_grad = time.perf_counter()
                    grad_time = (end_grad - start_grad) * 1000
                    
                    st.image(cam_image, use_container_width=True, caption=f"Heatmap (Calcul : {grad_time:.1f}ms)")
                except Exception as e:
                    st.error("Erreur Grad-CAM")
                finally:
                    model_pt.model.eval()

    # --- PARTIE DROITE : SCORES ---
    with main_col2:
        st.markdown("### 📊 Résultats")
        st.info(f"**Top : {classes[top_idx]}** ({probs[top_idx]*100:.1f}%)")
        
        # Affichage discret du temps d'inférence ONNX
        st.markdown(f"<p class='inference-text'>⏱️ Inférence ONNX : {onnx_time:.2f} ms</p>", unsafe_allow_html=True)
        
        sorted_indices = np.argsort(probs)[::-1]
        st.markdown("---")
        
        for i in sorted_indices:
            st.markdown(f"**{classes[i]}** : {probs[i]*100:.2f}%")
            st.progress(float(probs[i]))