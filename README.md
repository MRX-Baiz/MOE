# MOE: Bilingual Student Answer Grading

A Python prototype for automated student answer grading in French and Arabic. A deterministic language router selects a local Transformer classifier, with an optional TF-IDF/SVM fallback. A Streamlit interface accepts a question, reference answer, and student answer and displays one of three labels: `correct`, `partiel`, or `incorrect`.

## Repository status

This repository contains inference code, a web interface, prepared datasets, and notebooks summarizing historical metrics. **Trained models, feature vectorizers, data preparation scripts, and training/evaluation pipelines are not included.** A fresh clone can run the routing tests and open the interface, but cannot grade answers until compatible model artifacts are supplied.

The notebooks contain manually entered figures, including 93.2% for the French expert. They are historical summaries, not reproducible evaluation results. This repository does not establish production deployment, grading reliability, or measured inference latency.

## Architecture

1. The router checks the question and student answer for characters in `U+0600–U+06FF`. If any are present, it selects Arabic; otherwise it selects French.
2. The selected expert receives the question and reference answer as context, paired with the student answer. Inputs are truncated to 256 tokens.
3. If the expert is unavailable, the application tries a local TF-IDF/SVM model using the question and student answer.
4. If no compatible model is available, the application returns an explicit error.

“Mixture of Experts” is the project's name for language-based model selection. Routing uses a fixed rule; it is not a learned gating network or a sparse Transformer MoE. Both experts are loaded into memory when available. Inference currently selects Apple MPS when available, otherwise CPU.

## Files

| Path | Purpose |
| --- | --- |
| `predictor.py` | Text preprocessing, routing, model loading, and inference |
| `app_interface.py` | Streamlit grading interface |
| `test_app.py` | Automated routing and missing-artifact checks |
| `data/final/` | Prepared French, Arabic, bilingual, and English dataset files |
| `model_evolution_tracker.ipynb` | Plots of manually entered historical accuracy figures |
| `surviving_models_metrics.ipynb` | Tables of reported model specifications and metrics |
| `RESEARCH_PAPER.md` | Implementation notes and evaluation limitations |

## Setup

```bash
git clone https://github.com/MRX-Baiz/MOE.git
cd MOE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `venv\Scripts\activate`.

The dependency list is an unpinned project environment; it is not a verified lockfile.

### Model artifacts

To enable neural inference, provide local Hugging Face sequence-classification checkpoints and their tokenizer files at:

```text
models/expert_fr/final/
models/expert_ar/final/
```

The loader expects safetensors weights and three output classes in this order: `correct`, `partiel`, `incorrect`. Verify this mapping against the checkpoint's training configuration before using predictions.

The optional SVM fallback requires both:

```text
models/svm_tfidf_model_optimized.pkl
data/features/tfidf_vectorizer.pkl
```

These must be a matching classifier/vectorizer pair with compatible feature dimensions and numeric class labels `0`, `1`, `2` in the same order. Only load pickle files from a trusted source. No artifact download is currently provided.

### Run

```bash
streamlit run app_interface.py
```

Open `http://localhost:8501`, enter a question, a reference answer, and a student answer, then select **Analyser & Noter**. The reference is optional in the current implementation; when omitted, the question is reused as context. Neural output scores are uncalibrated softmax scores. The SVM fallback does not report confidence.

### Tests

```bash
python3 -m unittest -v test_app
```

These tests exercise routing, fallback selection, and missing-model behavior without trained weights. They do not validate grading accuracy.

## Dataset and evaluation limitations

The bilingual split contains 5,510 training, 1,181 validation, and 1,181 test question records. Each record contains multiple labeled student answers; these are not answer counts.

An audit of the committed JSON files found shared `(language, question, reference_answer)` keys across splits: 3 between train/validation, 4 between train/test, and 2 between validation/test. Context passages also overlap. Of 1,181 test answers labeled `correct`, 992 exactly match the reference answer. These properties need to be addressed when designing an evaluation of unseen questions and natural student responses.

Dataset generation, annotation provenance, original source versions, and redistribution permissions are not documented here. File names alone do not establish provenance or licensing. The English data is included, but the router has no English expert.

Before relying on reported accuracy, restore the training/evaluation pipeline and publish split construction, model configuration, seeds, per-language class metrics, and predictions on a held-out set. Include paraphrases, mixed-language input, and independently labeled student answers. Human review remains necessary for consequential grading decisions.
