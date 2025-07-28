FROM python:3.10-slim

WORKDIR /app

# Copy all files into container
COPY main.py /app/main.py
COPY models /app/models
COPY Collection1 /app/Collection1

# Install dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    joblib \
    nltk \
    sentence-transformers \
    PyMuPDF \
    tqdm \
    catboost

# Download nltk stopwords
RUN python -m nltk.downloader stopwords

# Pre-download model (single-line command, no indentation issues)
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); print('Model downloaded')"

# Force offline mode
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

CMD ["python", "main.py"]