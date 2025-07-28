import os
import json
import fitz
import string
from nltk.corpus import stopwords
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datetime import datetime
import numpy as np

# ===============================
# 🔹 Utility: Join lines
# ===============================
def smart_join_lines(lines):
    joined = []
    for i, line in enumerate(lines):
        if i == 0:
            joined.append(line)
        else:
            if joined[-1].endswith(' ') or line.startswith(' '):
                joined.append(line)
            else:
                joined.append(' ' + line)
    return ''.join(joined).strip()

# ===============================
# 1️⃣ Extract Features from PDF
# ===============================
def extract_features_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_font_sizes_doc = []
    all_text_blocks_raw = []
    stop_words = set(stopwords.words('english'))

    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        page_width = page.rect.width
        blocks = page.get_text("dict")['blocks']

        # Font sizes for the page
        page_font_sizes = []
        for b in blocks:
            if b['type'] == 0:
                for line in b['lines']:
                    for span in line['spans']:
                        page_font_sizes.append(round(span['size'], 2))
        max_font_size_page = max(page_font_sizes) if page_font_sizes else 1.0

        page_blocks = []
        for b_idx, b in enumerate(blocks):
            if b['type'] != 0:
                continue

            # Skip tables
            is_table_block = False
            if len(b['lines']) > 1:
                x_coords = [line['spans'][0]['origin'][0] for line in b['lines'] if line['spans']]
                if len(set(round(x, 1) for x in x_coords)) > 1:
                    is_table_block = True
            if is_table_block:
                continue

            # Extract text lines
            line_texts = []
            underline_present = False
            for line in b['lines']:
                text_line = ""
                for span in line['spans']:
                    text_line += span['text']
                    if span['flags'] & 4:
                        underline_present = True
                if text_line.strip():
                    line_texts.append(text_line)

            if not line_texts:
                continue

            combined_text = smart_join_lines(line_texts)

            first_span = b['lines'][0]['spans'][0]
            font_size = round(first_span['size'], 2)
            is_bold = bool(first_span['flags'] & 16)

            all_font_sizes_doc.append(font_size)

            block_data = {
                "text": combined_text,
                "font_size": font_size,
                "bbox": b['bbox'],
                "page_num": page_num,
                "page_height": page_height,
                "page_width": page_width,
                "max_font_size_page": max_font_size_page,
                "is_bold": is_bold,
                "is_underlined": underline_present,
                "block_index": b_idx,
            }
            all_text_blocks_raw.append(block_data)
            page_blocks.append(block_data)

        # Add spacing info
        for i, block in enumerate(page_blocks):
            block['space_above'] = (block['bbox'][1] - (page_blocks[i-1]['bbox'][3] if i > 0 else 0))
            block['space_below'] = ((page_blocks[i+1]['bbox'][1] if i < len(page_blocks)-1 else page_height) - block['bbox'][3])

    max_font_size_pdf = max(all_font_sizes_doc) if all_font_sizes_doc else 1.0

    processed_data = []
    for block in all_text_blocks_raw:
        text = block['text']
        words = text.strip().split()
        font_size_relative_to_max_pdf = block['font_size'] / max_font_size_pdf
        font_size_relative_to_max_page = block['font_size'] / block['max_font_size_page']
        punctuation_char = text.strip()[-1] if text.strip() and text.strip()[-1] in string.punctuation else "NULL"

        processed_data.append({
            "text": text,
            "font_size_relative_to_max_pdf": font_size_relative_to_max_pdf,
            "font_size_relative_to_max_page": font_size_relative_to_max_page,
            "is_bold": block['is_bold'],
            "num_words": len(words),
            "punctuation": punctuation_char,
            "x_pos_relative": block['bbox'][0] / block['page_width'],
            "y_pos_relative": block['bbox'][1] / block['page_height'],
            "page_no": block['page_num'],
            "title_case_ratio": sum(1 for w in words if w and w[0].isupper()) / len(words) if words else 0,
            "stopword_ratio": sum(1 for w in words if w.lower() in stop_words) / len(words) if words else 0,
            "space_above": block['space_above'] / block['page_height'],
            "space_below": block['space_below'] / block['page_height'],
            "is_underlined": block['is_underlined']
        })
    doc.close()
    return processed_data

# ===============================
# 2️⃣ Predict Headings
# ===============================
def predict_and_save_json_with_title(input_data, model_path, output_path):
    model = joblib.load(model_path)
    df = pd.DataFrame(input_data)

    # One-hot for punctuation
    df['punctuation'] = df['punctuation'].astype(str)
    df = pd.get_dummies(df, columns=['punctuation'], drop_first=False)

    expected_cols = getattr(model, "feature_names_", None) or getattr(model, "feature_names_in_", None)
    if expected_cols is None:
        raise AttributeError("Model has no feature_names_ or feature_names_in_ attributes.")

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]

    if df.isnull().values.any():
        df = pd.DataFrame(SimpleImputer(strategy='constant', fill_value=0).fit_transform(df), columns=df.columns)

    CLASS_NAMES = ['H1', 'H2', 'H3', 'None', 'Title']
    predictions = model.predict(df)

    pred_levels = [CLASS_NAMES[int(np.ravel([p])[0])] for p in predictions]
    outline = [{"level": level, "text": item["text"], "page": item["page_no"]}
               for item, level in zip(input_data, pred_levels) if level not in ["None", "Title"]]

    title_text = " ".join(item["text"] for item, level in zip(input_data, pred_levels) if level == "Title")
    final_json = {"title": title_text.strip(), "outline": outline}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=4)

# ===============================
# 3️⃣ Final Section Ranker
# ===============================
class FinalSectionRanker:
    def __init__(self, json_dir, pdf_dir):
        self.json_dir = json_dir
        self.pdf_dir = pdf_dir
        model_path = "/app/models/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_path if os.path.exists(model_path) else 'all-MiniLM-L6-v2')

    def extract_text_for_heading(self, doc, heading, next_heading=None):
        start_page = heading['page'] - 1
        end_page = (next_heading['page'] - 1) if next_heading else (len(doc) - 1)
        full_text = ""
        heading_found = False
        for page_num in range(start_page, end_page + 1):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    line_text = "".join(span["text"] for span in line["spans"])
                    if not heading_found and heading['text'].lower() in line_text.lower():
                        heading_found = True
                        continue
                    if heading_found:
                        if next_heading and next_heading['text'].lower() in line_text.lower():
                            return full_text.strip()
                        full_text += line_text + " "
        return full_text.strip()

    def load_sections_with_content(self):
        sections = []
        processed_docs = set()
        for filename in tqdm(os.listdir(self.json_dir)):
            if filename.endswith('.json'):
                pdf_filename = filename.replace('.json', '.pdf')
                pdf_path = os.path.join(self.pdf_dir, pdf_filename)
                if not os.path.exists(pdf_path):
                    continue
                processed_docs.add(pdf_filename)
                with open(os.path.join(self.json_dir, filename), 'r') as f:
                    data = json.load(f)
                doc = fitz.open(pdf_path)
                headings = data['outline']
                for i in range(len(headings)):
                    curr = headings[i]
                    nxt = headings[i+1] if i < len(headings)-1 else None
                    content = self.extract_text_for_heading(doc, curr, nxt)
                    if content:
                        sections.append({
                            "document": pdf_filename,
                            "section_title": curr['text'],
                            "page_number": curr['page'],
                            "level": curr['level'],
                            "content": content
                        })
                doc.close()
        return sections, list(processed_docs)

    def rank_sections(self, persona, job_description, top_n=10):
        sections, input_docs = self.load_sections_with_content()
        query = f"{persona}. {job_description}"
        section_texts = [f"{s['section_title']}. {s['content']}" for s in sections]
        embeddings = self.model.encode([query] + section_texts)
        similarities = cosine_similarity([embeddings[0]], embeddings[1:])[0]
        for i, s in enumerate(sections):
            s['similarity_score'] = float(similarities[i])
        ranked = sorted(sections, key=lambda x: x['similarity_score'], reverse=True)
        return {
            "metadata": {
                "input_documents": input_docs,
                "persona": persona,
                "job_to_be_done": job_description,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {"document": s['document'], "section_title": s['section_title'], "importance_rank": i+1, "page_number": s['page_number']}
                for i, s in enumerate(ranked[:top_n])
            ],
            "subsection_analysis": [
                {"document": s['document'], "refined_text": s['content'], "page_number": s['page_number']}
                for s in ranked[:10]
            ]
        }

# ===============================
# 🚀 MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    BASE_DIR = "/app/Collection1"
    PDF_DIR = os.path.join(BASE_DIR, "PDFs")
    INPUT_JSON = os.path.join(BASE_DIR, "challenge1b_input.json")
    OUTPUT_JSON = os.path.join(BASE_DIR, "challenge1b_output.json")
    MODEL_PATH = "/app/models/catboost_smote_model.joblib"

    with open(INPUT_JSON, "r") as f:
        config = json.load(f)
    persona = config.get("persona", "")
    job_desc = config.get("job_to_be_done", "")

    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            features = extract_features_from_pdf(pdf_path)
            json_path = os.path.join(PDF_DIR, pdf_file.replace(".pdf", ".json"))
            predict_and_save_json_with_title(features, MODEL_PATH, json_path)

    final_ranker = FinalSectionRanker(PDF_DIR, PDF_DIR)
    ranked_output = final_ranker.rank_sections(persona, job_desc)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(ranked_output, f, indent=2)

    print(f"✅ challenge1b_output.json saved at {OUTPUT_JSON}")
