# 🧠 MOE - Mixture of Experts (Bilingual ASAG System)

An Automated Student Answer Grading (ASAG) system built with a **Mixture of Experts (MoE)** architecture. This project uses dedicated advanced deep learning models to instantly and accurately grade student answers in both **French** and **Arabic**.

---

## 🚀 Features

*   **Bilingual Intelligence**: Specialized neural experts trained independently for French (`expert_fr`) and Arabic (`expert_ar`).
*   **Automatic Routing**: A smart language router automatically detects the language (via Unicode matching) and routes the question/answer to the correct expert.
*   **High Accuracy**: Achieves ~93.2% accuracy by understanding semantic meaning and synonyms, rather than just exact keyword matching.
*   **Interactive Interface**: A ready-to-use Streamlit web application for real-time grading and confidence scoring.
*   **Optimized Fallback**: Includes a robust, optimized TF-IDF + SVM baseline fallback model.

## 📁 Project Structure

```bash
MOE/
│
├── app_interface.py                  # The main Streamlit web application
├── test_app.py                       # Programmatic test script for the MoE Router
├── requirements.txt                  # Python dependencies
│
├── surviving_models_metrics.ipynb    # Notebook detailing the architecture and accuracy of the 3 final models
├── model_evolution_tracker.ipynb     # Notebook visualizing the accuracy growth across project milestones
│
├── models/                           # 🧠 The trained "Brains"
│   ├── expert_ar/final/              # The tuned Arabic MiniLM specialist
│   ├── expert_fr/final/              # The tuned French MiniLM specialist
│   └── svm_tfidf_model_optimized.pkl # The fallback classical model
│
└── data/
    ├── final/                        # Cleaned training, testing, and validation records
    └── features/                     # Pre-calculated TF-IDF vectorizers for the fallback
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MRX-Baiz/MOE.git
   cd MOE
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to Use (Running the App)

The easiest way to interact with the models is through the web interface.

1. Ensure your virtual environment is activated.
2. Run the Streamlit application:
   ```bash
   streamlit run app_interface.py
   ```
3. Open your browser to `http://localhost:8501`.
4. Enter the **Question**, the **Reference Answer**, and the **Student's Answer**.
5. Click "Analyser & Noter" to instantly see if the answer is **Correct**, **Partial**, or **Incorrect**, along with the confidence score and which Expert was used!

## 🧪 Testing the Routing Programmatically

If you just want to verify that the core Python logic successfully routes Arabic to the Arabic Expert and French to the French Expert without booting the UI, run the test script:

```bash
python test_app.py
```

## 📊 Analytics & Metrics

If you are interested in the data science behind this project, we have included two Jupyter Notebooks:
*   `model_evolution_tracker.ipynb`: Plots how the accuracy improved by over 20% by migrating from Classical ML to Neural MoE.
*   `surviving_models_metrics.ipynb`: Compares the architectural sizes and final evaluation metrics of the 3 models actively powering the application today.

---
*Built to improve grading speed and consistency in bilingual educational environments.*
