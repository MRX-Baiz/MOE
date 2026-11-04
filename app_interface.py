import streamlit as st

from predictor import MoEPredictor

# ==========================================
# 3. Streamlit Interface
# ==========================================
st.set_page_config(page_title="Bilingual ASAG MoE")

st.title("MoE ASAG: Système de Notation")
st.markdown("Automated Student Answer Grading — **Bilingual Mixture of Experts**")

@st.cache_resource
def get_moe(): return MoEPredictor()

moe = get_moe()

# Sidebar: Expert Status
st.sidebar.header("Mixture of Experts Status")
for lang, status in moe.expert_status.items():
    if status:
        st.sidebar.success(f"Expert {lang.upper()} Ready")
    else:
        st.sidebar.error(f"Expert {lang.upper()} Missing")

if not any(moe.expert_status.values()) and not moe.svm_available:
    st.warning("Aucun modèle local disponible. Consultez le README pour les fichiers requis.")

# Main Interface
with st.form("grading_form"):
    col1, col2 = st.columns(2)
    with col1:
        question = st.text_area("Question", height=100, placeholder="Ex: Que signifie le terme 'Cloud Computing'?")
    with col2:
        ref_answer = st.text_area("Réponse de Référence", height=100, placeholder="Optionnel: La réponse attendue...")
        
    student_answer = st.text_area("Réponse de l'Étudiant", height=150, placeholder="Entrez la réponse de l'étudiant ici...")
    
    submitted = st.form_submit_button("Analyser & Noter")

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
                col_b.metric("Score du modèle", f"{conf:.2%}" if conf is not None else "Indisponible")
                st.caption("Le score du modèle n’est pas une probabilité calibrée de justesse.")
                
                st.info(f"**Info**: Utilisé **{expert}**.")
                
                with st.expander("Voir les métriques détaillées"):
                    st.json(result)
