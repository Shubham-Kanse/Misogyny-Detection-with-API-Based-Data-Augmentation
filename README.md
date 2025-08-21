# Misogyny Detection with API-Based Data Augmentation

This project implements a real-time misogyny detection system with **continuous learning**.  
It is based on the MSc thesis *“Misogyny Detection with API-Based Data Augmentation”* (University of Galway, 2025).

The system uses **BERT-based models** and an **API-driven lifecycle** for:
- Detecting misogynistic content in text  
- Logging low-confidence predictions  
- Generating synthetic data through augmentation  
- Retraining models on enriched datasets  

---

## 🔑 Features
- **Prediction API (`/predict`)**: Classifies text as *misogynistic* or *non-misogynistic* with confidence scores.  
- **Augmentation API (`/augment`, `/augment/preview`)**: Generates synthetic variants using:
  - Synonym replacement  
  - Typo/noise injection  
  - Prompt-based LLM generation  
  - Label verification  
- **Retraining API (`/retrain`)**: Fine-tunes models on original + augmented datasets, producing new model versions.  
- **Continuous Loop**: Low-confidence inputs trigger augmentation → retraining → improved detection.  

---

## 📂 Project Structure
```
├── api/                  # FastAPI routes
├── services/             # Core services (predictor, augmentor, retrainer, model loader)
├── data/                 
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned and prepared datasets
│   ├── augmented/        # Synthetic datasets (v1, v2, ...)
├── models/               # Saved model checkpoints (misogyny_model_vN.pt)
├── notebooks/            # Experimental notebooks
├── retraining_pipeline/  # Training scripts
├── utility/              # Helper scripts
├── config.py             # Global configuration
├── requirements.txt      # Dependencies
└── main.py               # Entry point for FastAPI server
```

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/misogyny-detection-api.git
   cd misogyny-detection-api
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set environment variables in `.env` for configuration overrides.

---

## 🚀 Running the API
Start the FastAPI server with:
```bash
uvicorn main:app --reload
```

Open the interactive docs at:  
👉 http://127.0.0.1:8000/docs  

---

## 📌 Example Usage

### 1. Predict
```bash
curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{"text": "Women can’t drive properly"}'
```

### 2. Augment (Preview)
```bash
curl -X POST "http://127.0.0.1:8000/augment/preview"      -H "Content-Type: application/json"      -d '{"text": "She should stay in the kitchen"}'
```

### 3. Retrain
```bash
curl -X POST "http://127.0.0.1:8000/retrain"
```

---

## 📚 References
- Kanse, Shubham (2025). *Misogyny Detection with API-Based Data Augmentation*. MSc Thesis, University of Galway.  
- Mohasseb et al. (2023). *Augmented NLP for Misogyny Detection*.  

---

## 📝 License
This project is for **academic and research purposes only**. Not intended for production use without further validation.  
