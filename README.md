# Retinal XAI — Cardiovascular Risk Prediction

**Project Title:** Explainable AI for Retinal Image Analysis & Cardiovascular Risk Prediction

This project is a research prototype web application that analyses a retinal fundus image and produces a Diabetic Retinopathy (DR) grade, a Cardiovascular Disease (CVD) risk score, a fused risk level, and explainable AI insights (Grad-CAM heatmaps and an LLM-generated clinical report).

> ⚠️ **RESEARCH DISCLAIMER:** This is an academic prototype only. It does NOT provide medical diagnoses.

## 🚀 Features

1. **Diabetic Retinopathy (DR) Grading:** Classifies DR (0–4) using a fine-tuned ResNet-50 model.
2. **Cardiovascular Disease (CVD) Risk Score:** Predicts binary CVD risk (0.0–1.0) using a fine-tuned EfficientNet-B4 model.
3. **Fused Risk Level:** Combines DR and CVD scores into a holistic risk level (Low / Moderate / High), with clinical safety guardrails.
4. **Grad-CAM Heatmaps:** Visualises the exact retinal regions the AI focused on for explainability.
5. **Natural Language AI Report:** Generates a structured clinical report using BioMistral-7B via Ollama.
6. **Interactive Chatbot:** Allows users to ask follow-up questions about the analysis results safely.
7. **Patient History Integration:** Factors in clinical history (text) to adjust the final risk score.
8. **Prediction History:** Stores and displays past predictions using a MySQL database.

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Backend** | Python 3.10, Flask | Web server and API routing |
| **Database** | MySQL + SQLAlchemy | Persistent storage of predictions |
| **DR Model** | ResNet-50 (PyTorch) | Classify DR Grade (5 classes) |
| **CVD Model** | EfficientNet-B4 (PyTorch) | Predict CVD Risk (Binary) |
| **Explainability** | Grad-CAM, LSTM Captioner | Visual heatmaps and text features |
| **LLM Engine** | Mistral-7B (BioMistral) / Ollama | Natural language generation |
| **Hardware Accel.** | Apple M3 MPS Backend | Local GPU acceleration on macOS |
| **Frontend** | HTML, CSS, Vanilla JS | Interactive Web UI |

## 🧠 Machine Learning Models

- **ResNet-50 (DR):** Trained on the APTOS 2019 dataset (3,662 images). Achieved an AUC of 0.9338.
- **EfficientNet-B4 (CVD):** Trained on the ODIR-5K dataset (12,460 images). Achieved an AUC of 0.9594.
- **LSTM Captioner:** Translates visual features into clinical captions, trained via Label-to-Caption templating.
- **Fusion Logic:** Score-level fusion combining DR severity and direct CVD risk, heavily weighted towards direct CVD indicators and adjusted with a clinical safety multiplier (`1.25x`) to prevent false negatives.

## ⚙️ How It Works (Pipeline)

1. **Preprocess:** Resize (224×224), normalise, and convert image to float32 tensor.
2. **CNN Inference:** Run ResNet-50 (DR) and EfficientNet-B4 (CVD) to get raw predictions.
3. **Fusion:** Combine scores to get a holistic Risk Level (Low/Moderate/High).
4. **Grad-CAM:** Generate a colour heatmap overlay showing the CNN's focal points.
5. **LLM Generation:** Inject the scores, caption, and clinical history into a BioMistral prompt. The LLM generates the report.
6. **Database Save:** Commit all data to the MySQL database.
7. **Results Page:** Render the complete analysis to the user.

## 💻 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/felixkevinsinghuk-png/final-year-project.git
   cd final-year-project
   ```

2. **Set up the Conda environment:**
   ```bash
   conda create -n retinal_xai python=3.10
   conda activate retinal_xai
   pip install -r requirements.txt
   ```

3. **Install & Start Dependencies:**
   - Install MySQL and create the required database.
   - Install [Ollama](https://ollama.com/) and pull the BioMistral model:
     ```bash
     ollama pull biomistral
     ```

## 🚀 Running the Application

1. **Start MySQL:**
   ```bash
   brew services start mysql
   ```

2. **Start Ollama Server:**
   ```bash
   # Adjust path to where your models are stored if necessary
   OLLAMA_MODELS="/path/to/ollama_models" ollama serve &
   ```

3. **Run the Flask App:**
   ```bash
   python app.py
   ```

4. **Access the Web Interface:**
   Open your browser and navigate to `http://localhost:5001`.

## 📚 Further Reading
For deeper insights into the project's architecture, ML methodologies, clinical safety features (like the false negative guardrails), and known limitations, please refer to the `Documentation.md` and `Technical_Decisions_and_Limitations.md` files included in this repository.
