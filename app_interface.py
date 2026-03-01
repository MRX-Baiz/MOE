import streamlit as st
import os
import json
import pickle
import numpy as np
import torch
from pathlib import Path
import re

# ==========================================
# 1. Preprocessing Utilities
# ==========================================
def clean_text(text, lang='fr'):
    if not text: return ""
    text = str(text).strip()
    if lang == 'ar':
        # Remove diacritics and non-arabic chars
        diacritics = ['\u064B', '\u064C', '\u064D', '\u064E', '\u064F', '\u0650', '\u0651', '\u0652', '\u0670', '\u0640']
        for d in diacritics: text = text.replace(d, '')
        text = re.sub(r'[^\u0600-\u06FF\s?.,!;:()\'-]', '', text)
    else:
        # French/English: basic punct cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s?.,!;:()\'-]', '', text)
    return text

# ==========================================
# 2. MoE Predictor Class (FR / AR Specialist)
# ==========================================
class MoEPredictor:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.models_dir = self.base_dir / "models"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load Experts
        self.experts = {}
        self.tokenizers = {}
        self.expert_status = {"fr": False, "ar": False}
        
        self._load_expert("fr", self.models_dir / "expert_fr" / "final")
        self._load_expert("ar", self.models_dir / "expert_ar" / "final")
        
        # SVM Fallback
        self._load_svm()

    def _load_expert(self, lang, path):
        if path.exists():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizers[lang] = AutoTokenizer.from_pretrained(path, use_fast=False)
                self.experts[lang] = AutoModelForSequenceClassification.from_pretrained(path, use_safetensors=True)
                self.experts[lang].to(self.device).eval()
                self.expert_status[lang] = True
            except Exception as e:
                st.error(f"Error loading {lang.upper()} Expert: {e}")

    def _load_svm(self):
        self.svm_available = False
        svm_path = self.models_dir / "svm_tfidf_model_optimized.pkl"
        vec_path = self.base_dir / "data" / "features" / "tfidf_vectorizer.pkl"
        if svm_path.exists() and vec_path.exists():
            with open(svm_path, 'rb') as f: self.svm_model = pickle.load(f)
            with open(vec_path, 'rb') as f: self.tfidf_vectorizer = pickle.load(f)
            self.svm_available = True

    def route(self, text):
        # Unicode detection: Arabic chars in text?
        if re.search(r'[\u0600-\u06FF]', text):
            return "ar"
        return "fr"

    def predict(self, question, student_answer, ref_answer=""):
        lang = self.route(question + student_answer)
        
        if self.expert_status.get(lang):
            return self._predict_expert(lang, question, student_answer, ref_answer)
        elif self.svm_available:
            return self._predict_svm(question, student_answer, lang)
        else:
            return {"error": "No model specialized for this language found."}

    def _predict_expert(self, lang, question, student_answer, ref_answer):
        model = self.experts[lang]
        tokenizer = self.tokenizers[lang]
        
        # Contextual Input (Matching 14_moe_training.ipynb)
        # Use provided ref_answer if available, otherwise fallback to question
        ref = ref_answer if ref_answer else question
        sep = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"
        
        context = f"{clean_text(question, lang)} {sep} {clean_text(ref, lang)}"
        student = clean_text(student_answer, lang)
        
        inputs = tokenizer(context, student, truncation=True, padding=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            
        labels = ['correct', 'partiel', 'incorrect']
        return {
            'prediction': labels[pred_idx],
            'confidence': float(probs[pred_idx]),
            'language': lang,
            'expert_used': f"Specialist Expert ({lang.upper()})",
            'accuracy_estimate': "93.2%" if lang == "fr" else "90%+"
        }

    def _predict_svm(self, question, student_answer, lang):
        q_proc = clean_text(question, lang)
        a_proc = clean_text(student_answer, lang)
        combined = f"{q_proc} {a_proc}"
        vector = self.tfidf_vectorizer.transform([combined])
        pred_idx = self.svm_model.predict(vector)[0]
        labels = ['correct', 'partiel', 'incorrect']
        return {
            'prediction': labels[pred_idx],
            'confidence': 0.85, # SVM estimate
            'language': lang,
            'expert_used': "SVM Baseline Fallback"
        }

# ==========================================
# 3. Streamlit Interface
# ==========================================
st.set_page_config(page_title="Bilingual ASAG MoE", page_icon="🧠")

st.title("🧠 MoE ASAG: Système de Notation")
st.markdown("Automated Student Answer Grading — **Bilingual Mixture of Experts**")

@st.cache_resource
def get_moe(): return MoEPredictor()

moe = get_moe()

# Sidebar: Expert Status
st.sidebar.header("Mixture of Experts Status")
for lang, status in moe.expert_status.items():
    if status:
        st.sidebar.success(f"✅ Expert {lang.upper()} Ready")
    else:
        st.sidebar.error(f"❌ Expert {lang.upper()} Missing")

# Main Interface
with st.form("grading_form"):
    col1, col2 = st.columns(2)
    with col1:
        question = st.text_area("Question", height=100, placeholder="Ex: Que signifie le terme 'Cloud Computing'?")
    with col2:
        ref_answer = st.text_area("Réponse de Référence", height=100, placeholder="Optionnel: La réponse attendue...")
        
    student_answer = st.text_area("Réponse de l'Étudiant", height=150, placeholder="Entrez la réponse de l'étudiant ici...")
    
    submitted = st.form_submit_button("📝 Analyser & Noter")

if submitted:
    if not question or not student_answer:
        st.warning("Veuillez remplir la Question et la Réponse de l'étudiant.")
    else:
        with st.spinner('L’expert analyse la réponse...'):
            result = moe.predict(question, student_answer, ref_answer)
            
            if "error" in result:
                st.error(result['error'])
            else:
                grade = result['prediction']
                conf = result['confidence']
                lang = result['language'].upper()
                expert = result['expert_used']
                
                # Visual Feedback
                color = "green" if grade == "correct" else "orange" if grade == "partiel" else "red"
                st.markdown(f"### Score: <span style='color:{color}'>{grade.upper()}</span>", unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                col_a.metric("Expert", lang)
                col_b.metric("Confiance", f"{conf:.2%}")
                
                st.info(f"💡 **Info**: Utilisé **{expert}**.")
                
                with st.expander("Voir les métriques détaillées"):
                    st.json(result)
