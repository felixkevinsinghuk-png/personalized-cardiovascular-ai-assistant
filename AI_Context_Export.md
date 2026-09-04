# Retinal XAI: Complete Technical Context for AI Assistants
*Use this file to copy-paste into Claude, ChatGPT, or Gemini when you need to give them full context of your project.*

## 1. Project Overview & Architecture
*   **Goal:** A multi-modal AI system to assess Diabetic Retinopathy (DR) and Cardiovascular Disease (CVD) risk from a single retinal fundus image, accompanied by an LLM-generated clinical report and an interactive chatbot.
*   **Hardware:** Apple MacBook Air (M3, 8GB RAM), running PyTorch via the `mps` (Metal Performance Shaders) backend.
*   **Tech Stack:** Python 3.10, Flask (Port 5001), PyTorch, MySQL, Ollama (Mistral-7B).

### The Pipeline
1.  **Input:** Retinal image uploaded via web interface.
2.  **Preprocessing:** Resized to 224x224, normalised using ImageNet means/stds.
3.  **DR Inference:** ResNet-50 predicts DR Grade (0 to 4).
4.  **CVD Inference:** EfficientNet-B4 predicts CVD Risk Score (0.0 to 1.0).
5.  **Fusion:** The two scores are combined using a weighted formula with a clinical safety multiplier.
6.  **Explainability:** Grad-CAM generates a heatmap highlighting influential retinal regions based on ResNet-50 layer4.
7.  **LLM Reporting:** PromptBuilder injects all scores into a strict prompt. Mistral-7B (via Ollama) generates a natural language report. A Safety Filter ensures no non-medical questions are answered.
8.  **Database:** Results, heatmaps, and chat history are committed to MySQL.

---

## 2. Model Parameters & Metrics

### Model A: ResNet-50 (Diabetic Retinopathy)
*   **Dataset:** APTOS 2019 (3,662 images). Highly imbalanced (Grade 0 has 9x more images than Grade 3).
*   **Architecture:** Pre-trained on ImageNet. Layers 1 & 2 frozen. FC Head: `Dropout(0.5) -> Linear(2048, 5)`.
*   **Loss Function:** Initially `CrossEntropyLoss`. (Tried Weighted Loss to fix imbalance, but it yielded negligible improvements).
*   **Test Metrics:** AUC: 0.9332 | F1-Score: 0.6465 | Sensitivity: 0.6362 | Specificity: 0.9520

### Model B: EfficientNet-B4 (Cardiovascular Risk)
*   **Dataset:** ODIR-5K (Ocular Disease Intelligent Recognition).
*   **Architecture:** Pre-trained on ImageNet. Fully fine-tuned. FC Head: `Dropout(0.3) -> Linear(1792, 1) -> Sigmoid`.
*   **Loss Function:** `BCEWithLogitsLoss`.
*   **Test Metrics:** AUC: 0.9675 | F1-Score: 0.9172 | Sensitivity: 0.9245 | Specificity: 0.9945

---

## 3. Fusion Logic & Risk Thresholds
A custom "Weighted Late Fusion" (Score-Level Fusion) algorithm combines the outputs.

### Formula
```python
normalised_dr = dr_grade / 4.0
raw_fused = (0.4 * normalised_dr) + (0.6 * cvd_score)
adjusted_fused = raw_fused * 1.25  # Clinical Safety Multiplier
```
*   **Why 0.6 for CVD?** EfficientNet-B4 directly predicts systemic risk and outputs a highly precise probability.
*   **Why 0.4 for DR?** DR severity is a secondary clinical proxy for CVD risk. Since the score is less granular (0, 0.25, 0.5, etc.), a lower weight prevents it from drowning out the precise CVD score.

### Risk Tiers (Post-Multiplier)
*   **Low Risk:** `< 0.25`
*   **Moderate Risk:** `0.25 to 0.67`
*   **High Risk:** `> 0.67`

---

## 4. Technical & Logical Issues Faced (and Fixed)

### Issue 1: False Negatives & Clinical Safety
*   **The Incident:** A retinal image of a patient with 14 years of diabetes and reverse blood flow in the heart (diagnosed 2 years later) yielded a "Low Risk" (0.0055) score.
*   **Root Cause Analysis:** 
    1. The image was 2 years old; visible retinal microvascular damage lags significantly behind the onset of systemic disease.
    2. Reverse blood flow is a *structural* valvular defect, which does not manifest as microvascular markers on the retina. 
*   **The Fix:** To heavily penalise false negatives (erring on the side of caution), a **Clinical Safety Multiplier (1.25x)** was added to the raw fused score, and the **Low Risk Threshold was lowered from 0.33 to 0.25**. Borderline patients are now pushed into Moderate Risk. Retraining the EfficientNet model with an asymmetric loss (`pos_weight`) was avoided to prevent destroying the model's exceptional 99.45% Specificity.

### Issue 2: Severe Class Imbalance in APTOS Dataset
*   **The Problem:** The ResNet-50 model achieved a low Macro F1-Score (0.6465) due to a massive lack of severe (Grade 3/4) DR images compared to Grade 0.
*   **The Fix:** Implemented `Weighted CrossEntropyLoss` with weights `[0.5, 2.0, 1.0, 3.5, 2.5]` to heavily penalise misclassification of rare grades.
*   **The Result:** Functionally identical performance (F1: 0.6483). Proved that simple loss-weighting is insufficient for extreme structural data imbalance. Future work requires SMOTE or GAN-based synthetic data generation.

### Issue 3: Hardware Limitations & LLM Integration
*   **The Problem:** The 4.1GB Mistral-7B LLM running via Ollama caused local timeouts (120s) when loading from the external SSD to the M3 RAM on the first query.
*   **The Fix:** Increased the `OLLAMA_TIMEOUT` to 300s in `config.py`.
*   **The Problem:** HuggingFace's MedGemma model failed with a 401 Gated Access error.
*   **The Fix:** Ripped out MedGemma and fully routed the application to rely entirely on the local Mistral-7B instance.

### Issue 4: AI Guardrails & Prompt Injection
*   **The Problem:** The Mistral chatbot would attempt to answer non-medical questions (e.g., recipes) by awkwardly weaving them into the clinical context.
*   **The Fix:** Implemented a `safety_filter.py` and strict prompt engineering instructions ordering the LLM to outright refuse non-biological/medical questions with a hardcoded disclaimer string.

### Issue 5: Environment & Port Conflicts
*   **The Problem:** Flask defaults to Port 5000, which on modern macOS is permanently occupied by the AirPlay Receiver service.
*   **The Fix:** Reconfigured the Flask application to run on Port 5001. Used `ngrok http 5001` to safely tunnel the localhost connection to the internet for external academic demonstrations without deploying to a limited-resource university cPanel.
