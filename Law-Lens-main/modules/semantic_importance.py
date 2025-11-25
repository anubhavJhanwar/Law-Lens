import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disable TensorFlow before importing

import time
import logging
from typing import List, Dict, Optional
from groq import Groq
from groq import RateLimitError, APIError
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Legal concept keywords for semantic filtering
LEGAL_CONCEPTS = [
    "contract", "agreement", "liability", "indemnify", "warranty", "breach",
    "obligation", "covenant", "clause", "provision", "party", "parties",
    "jurisdiction", "arbitration", "dispute", "remedy", "damages", "penalty",
    "termination", "confidential", "proprietary", "intellectual property",
    "compliance", "regulation", "statute", "law", "legal", "rights",
    "duties", "responsibilities", "enforce", "binding", "consent", "waiver"
]

# Retry configuration
MAX_RETRIES = 4
RETRY_DELAYS = [2, 4, 8, 15]  # Exponential backoff in seconds
BATCH_SIZE = 10  # Process 10 paragraphs per batch


def load_semantic_model() -> Optional[SentenceTransformer]:
    """Load sentence transformer for semantic understanding."""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logging.info("Semantic model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load semantic model: {e}")
        return None


def check_legal_relevance(paragraph: str, model: SentenceTransformer, threshold: float = 0.3) -> bool:
    """
    Check if paragraph discusses legal concepts using semantic similarity.
    Returns True if paragraph is legally relevant.
    """
    if not paragraph or not model:
        return False
    
    try:
        para_embedding = model.encode(paragraph, convert_to_tensor=False)
        concept_embeddings = model.encode(LEGAL_CONCEPTS, convert_to_tensor=False)
        
        para_norm = para_embedding / np.linalg.norm(para_embedding)
        concept_norms = concept_embeddings / np.linalg.norm(concept_embeddings, axis=1, keepdims=True)
        
        similarities = np.dot(concept_norms, para_norm)
        max_similarity = np.max(similarities)
        
        return max_similarity >= threshold
    except Exception as e:
        logging.error(f"Error in semantic check: {e}")
        return False


def _create_batch_prompt(paragraphs: List[str]) -> str:
    """Create a batched prompt for multiple paragraphs."""
    prompt = """You are a legal document analyzer. Analyze each paragraph below and rate its importance on a scale of 1-3:

3 = HIGH IMPORTANCE: Critical legal obligations, liabilities, rights, penalties, termination clauses, indemnification, warranties, or binding commitments
2 = MEDIUM IMPORTANCE: Supporting legal terms, definitions, procedural details, or contextual information
1 = LOW IMPORTANCE: General statements, background information, or non-binding content

Respond with ONLY the numbers (1, 2, or 3) separated by commas, in the same order as the paragraphs. Example: 3,2,1,3,2

Paragraphs:
"""
    for i, para in enumerate(paragraphs, 1):
        prompt += f"\n[{i}] {para[:500]}...\n"  # Limit to 500 chars per paragraph
    
    prompt += "\nYour response (numbers only, comma-separated):"
    return prompt


def _validate_batch_response(response_text: str, expected_count: int) -> Optional[List[int]]:
    """Validate and parse batch response from Groq."""
    try:
        response_text = response_text.strip()
        scores = [int(s.strip()) for s in response_text.split(',')]
        
        if len(scores) != expected_count:
            logging.warning(f"Expected {expected_count} scores, got {len(scores)}")
            return None
        
        if not all(score in [1, 2, 3] for score in scores):
            logging.warning(f"Invalid scores in response: {scores}")
            return None
        
        return scores
    except Exception as e:
        logging.error(f"Failed to parse batch response: {e}")
        return None


def _score_batch_with_retry(paragraphs: List[str]) -> List[int]:
    """
    Score a batch of paragraphs with exponential backoff retry logic.
    Returns list of scores (1-3) for each paragraph.
    """
    prompt = _create_batch_prompt(paragraphs)
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            response_text = response.choices[0].message.content.strip()
            scores = _validate_batch_response(response_text, len(paragraphs))
            
            if scores:
                return scores
            
            logging.warning(f"Invalid response format, attempt {attempt + 1}/{MAX_RETRIES}")
            
        except RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logging.warning(f"Rate limit hit. Retrying in {delay}s... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                logging.error(f"Rate limit exceeded after {MAX_RETRIES} attempts")
                return [1] * len(paragraphs)
        
        except APIError as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logging.warning(f"API error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                logging.error(f"API error after {MAX_RETRIES} attempts: {e}")
                return [1] * len(paragraphs)
        
        except Exception as e:
            logging.error(f"Unexpected error in batch scoring: {e}")
            return [1] * len(paragraphs)
    
    # Fallback if all retries failed
    logging.error("All retry attempts failed, returning default scores")
    return [1] * len(paragraphs)


def analyze_paragraphs_hybrid(paragraphs: List[str]) -> List[Dict]:
    """
    Hybrid analysis: semantic filtering + batched Groq scoring with retry logic.
    
    Returns list of:
    {
        "paragraph": str,
        "is_legal": bool,
        "importance_score": int (1-3),
        "importance": str ("high"/"medium"/"low")
    }
    """
    logging.info(f"Starting hybrid analysis for {len(paragraphs)} paragraphs...")
    
    semantic_model = load_semantic_model()
    if not semantic_model:
        logging.error("Semantic model failed to load, using fallback")
        return _fallback_analysis(paragraphs)
    
    # Step 1: Filter legally relevant paragraphs
    legal_paragraphs = []
    paragraph_map = {}  # Maps index to original paragraph
    
    for i, para in enumerate(paragraphs):
        if not para or len(para.strip()) < 40:
            continue
        
        is_legal = check_legal_relevance(para, semantic_model)
        
        if is_legal:
            legal_paragraphs.append(para)
            paragraph_map[len(legal_paragraphs) - 1] = i
    
    logging.info(f"Filtered to {len(legal_paragraphs)} legally-relevant paragraphs")
    
    # Step 2: Batch score legal paragraphs
    all_scores = []
    total_batches = (len(legal_paragraphs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(0, len(legal_paragraphs), BATCH_SIZE):
        batch = legal_paragraphs[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} paragraphs)...")
        
        scores = _score_batch_with_retry(batch)
        all_scores.extend(scores)
        
        # Rate limiting: small delay between batches
        if batch_idx + BATCH_SIZE < len(legal_paragraphs):
            time.sleep(0.5)
    
    # Step 3: Build results
    results = []
    score_idx = 0
    
    for i, para in enumerate(paragraphs):
        if not para or len(para.strip()) < 40:
            continue
        
        is_legal = check_legal_relevance(para, semantic_model)
        
        if is_legal and score_idx < len(all_scores):
            score = all_scores[score_idx]
            score_idx += 1
        else:
            score = 1
        
        importance = "high" if score == 3 else "medium" if score == 2 else "low"
        
        results.append({
            "paragraph": para,
            "is_legal": is_legal,
            "importance_score": score,
            "importance": importance
        })
    
    logging.info(f"Hybrid analysis complete. Processed {len(legal_paragraphs)} paragraphs via Groq API")
    return results


def _fallback_analysis(paragraphs: List[str]) -> List[Dict]:
    """Fallback to keyword-based analysis if semantic model fails."""
    logging.info("Using fallback keyword-based analysis")
    results = []
    
    for para in paragraphs:
        if not para or len(para.strip()) < 40:
            continue
        
        para_lower = para.lower()
        keyword_count = sum(1 for concept in LEGAL_CONCEPTS if concept in para_lower)
        
        if keyword_count >= 3:
            importance = "high"
            score = 3
        elif keyword_count >= 1:
            importance = "medium"
            score = 2
        else:
            importance = "low"
            score = 1
        
        results.append({
            "paragraph": para,
            "is_legal": keyword_count > 0,
            "importance_score": score,
            "importance": importance
        })
    
    return results
