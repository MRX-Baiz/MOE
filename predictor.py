import logging
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
                logging.getLogger(__name__).warning("Could not load %s expert: %s", lang, e)

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
        
        # Encode the question and reference as context for the student answer.
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
            'expert_used': f"Specialist Expert ({lang.upper()})"
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
            'confidence': None, # This fallback does not provide calibrated probabilities.
            'language': lang,
            'expert_used': "SVM Baseline Fallback"
        }

