def load_legalbert_model():
    # Import heavy libraries only when the model is actually needed.
    import os
    os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow before importing
    
    from transformers import AutoTokenizer, AutoModel  # type: ignore[import]
    from keybert import KeyBERT  # type: ignore[import]
    import torch  # type: ignore[import]

    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    kw_model = KeyBERT(model=model)
    return kw_model

# Extract legal keywords
def extract_legal_keywords(text, kw_model, top_n=10):
    if not text.strip():
        return []

    # Extract keywords using embeddings from LegalBERT
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n
    )

    return [kw[0] for kw in keywords]  # Only return the keyword text
