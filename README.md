# Resume Screener — ML-Powered Job Category Classifier

An end-to-end Machine Learning project that analyzes uploaded resumes (PDF or TXT) and predicts the most suitable **job category** using a trained text classification pipeline.

Built as an internship demonstration project with a clean **Streamlit** frontend.

---

## 🗂️ Project Structure

```
resume_screener/
├── app/
│   └── streamlit_app.py         # Streamlit frontend
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Resume dataset loading & preprocessing
│   ├── preprocessor.py          # Text cleaning & NLP preprocessing
│   ├── feature_engineering.py   # TF-IDF feature extraction
│   ├── model.py                 # ML model training & evaluation
│   └── predictor.py             # Inference on new resumes
├── models/
│   └── resume_pipeline.pkl      # Saved sklearn pipeline (after training)
├── data/
│   ├── raw/                     # Place raw CSV dataset here
│   └── processed/               # Auto-generated processed files
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
├── train.py                     # Run this to train & save the model
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
Download the **Resume Dataset** from Kaggle:
https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset

Place `UpdatedResumeDataSet.csv` inside `data/raw/`.

### 3. Train the model
```bash
python train.py
```
This will train a TF-IDF + Logistic Regression pipeline and save it to `models/resume_pipeline.pkl`.

### 4. Launch the app
```bash
streamlit run app/streamlit_app.py
```

---

## ML Pipeline

```
Raw Text (PDF/TXT)
      │
      ▼
Text Cleaning (lowercase, remove special chars, stopwords, lemmatization)
      │
      ▼
TF-IDF Vectorizer (max 5000 features, bigrams)
      │
      ▼
Logistic Regression Classifier
      │
      ▼
Predicted Job Category + Confidence Score
```

---

## Job Categories Supported

Java Developer, Python Developer, Data Science, Machine Learning, DevOps Engineer,
Web Designing, HR, Civil Engineer, Business Analyst, SAP Developer, Blockchain,
ETL Developer, Network Security, PMO, Database, Hadoop, DotNet Developer, Testing,
Mechanical Engineer, Sales, Operations Manager, Arts, Health and Fitness, Advocate

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn |
| NLP | NLTK |
| PDF Parsing | pdfplumber |
| Frontend | Streamlit |
| Serialization | joblib |

---

## Notes for Internship Presentation

- The project uses a **single sklearn Pipeline** object (`TfidfVectorizer` → `LogisticRegression`) ensuring no data leakage between train/test.
- All preprocessing is encapsulated in `src/preprocessor.py` making it reusable.
- The `predictor.py` module loads the saved pipeline and works on any new resume at inference time.
- Model metrics (accuracy, classification report, confusion matrix) are printed after training.
