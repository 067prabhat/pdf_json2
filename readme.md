# 📄 Adobe2 PDF Section Extraction & Ranking

This project processes PDF documents to extract headings, detect structure, and rank important sections based on a given persona and job description.  
It uses machine learning models to detect headings and a semantic similarity model to rank the most relevant sections.

---

## 🚀 **Approach**

1. **Feature Extraction from PDFs**  
   - Uses [PyMuPDF](https://pymupdf.readthedocs.io/) to read PDF blocks, font sizes, and styles.  
   - Extracts **font size ratios, bold text indicators, spacing, position**, and text characteristics as features.

2. **Heading Prediction**  
   - A **CatBoost model** (`catboost_smote_model.joblib`) predicts headings (H1, H2, H3) and the document title.  
   - Predictions are processed into a JSON outline.

3. **Section Ranking**  
   - [SentenceTransformers](https://www.sbert.net/) generates embeddings for headings and section content.  
   - Sections are ranked based on cosine similarity with the given persona and job description.

4. **Output**  
   - A ranked JSON output is generated at `Collection1/challenge1b_output.json`.

---

## 🧠 **Tech Stack**
- **Python 3.10**
- **Libraries**  
  - `PyMuPDF` (PDF parsing)  
  - `CatBoost` (heading classification model)  
  - `SentenceTransformers` (semantic similarity)  
  - `scikit-learn`, `pandas`, `numpy` (data preprocessing)  
  - `nltk` (stopword processing)  
  - `tqdm` (progress visualization)
- **Docker** (for containerized execution)

---

## 📂 **Project Structure**
```plaintext

│
├── main.py
├── models/
│   └── catboost_smote_model.joblib
├── Collection1/
│   ├── challenge1b_input.json   # Input configuration
│   ├── challenge1b_output.json  # Output (auto-generated)
│   ├── PDFs/
│   │   ├── document1.pdf
│   │   ├── document2.pdf
│   │   └── ...
└── Dockerfile

📥 Adding Input Files
/Collection1/PDFs/

Create the input JSON challenge1b_input.json inside:
/Collection1/


🛠 Building & Running the Solution
1️⃣ Build the Docker Image
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

2️⃣ Run the Container
docker run --rm \
  -v $(pwd)/Collection1:/app/Collection1 \
  --network none \
  mysolutionname:somerandomidentifier

  📤 Output
  /Collection1/challenge1b_output.json

