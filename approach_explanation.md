Approach Explanation
Our approach is a multi-stage pipeline designed to intelligently parse, understand, and rank sections from a collection of PDF documents based on a user's query. The process combines structural analysis with semantic understanding to deliver highly relevant results.

## 1. PDF Feature Extraction
The first step is to deconstruct each PDF into meaningful text blocks and extract a rich set of features. Instead of treating the document as a flat text file, we analyze its visual and structural properties, much like a human reader would. For each text block, we capture not just the text itself, but also metadata like:

Typographical Cues: Font size (both absolute and relative to the page/document), boldness, and underlining. These are strong indicators of headings.

Positional Data: The block's relative position (x, y) on the page and the spacing above and below it. Headings often have more whitespace around them.

Content-based Features: The number of words, the ratio of title-cased words, and the ratio of stopwords. Headings tend to be short, capitalized, and have fewer stopwords than paragraphs.

This comprehensive feature set allows us to create a detailed, structured representation of the PDF content.

## 2. Heading and Title Prediction
With the features extracted, we feed them into a pre-trained CatBoost machine learning model. This model's job is to classify each text block into one of five categories: Title, H1, H2, H3, or None (regular text). By doing this for every PDF, we effectively generate a structured outline or a table of contents for each document. This transforms the unstructured PDFs into organized JSON files, which are much easier to work with in the next stage.

## 3. Semantic Ranking of Sections
The final stage is where we find the most relevant information for the user. The process is as follows:

Content Extraction: We use the outlines generated in the previous step to extract the full text content under each identified heading. A "section" is defined as the heading's title plus all the text that follows it, up until the next heading.

Semantic Search: We combine the user's persona and job_to_be_done to form a query. Using a SentenceTransformer model (all-MiniLM-L6-v2), we convert this query and every extracted section into numerical vectors (embeddings) that capture their semantic meaning.

Similarity Scoring: We calculate the cosine similarity between the query's vector and each section's vector. This gives us a score from -1 to 1, indicating how semantically similar the section's content is to the user's query.

Ranking: Finally, we rank all the sections across all documents in descending order of their similarity scores and return the top results. This ensures the most relevant sections appear first, regardless of which document they came from.