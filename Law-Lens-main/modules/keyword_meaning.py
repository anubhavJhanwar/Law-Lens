import os
import json  # <-- Import json
from groq import Groq

# Initialize the client (make sure GROQ_API_KEY is set in your environment)
try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Warning: Could not initialize Groq client. {e}")
    client = None

def get_keywords_meaning_smart(keywords: list[str]) -> dict:
    """
    Use Groq API (Llama) to decide which keywords need explanation and explain them briefly.
    Returns a dictionary: {keyword: meaning or "No explanation needed"}
    """
    if not client:
        return {"error": "Groq client is not initialized. Check GROQ_API_KEY."}
        
    if not keywords:
        return {} # Return an empty dict if no keywords are provided

    # Combine all keywords into one string for the prompt
    keyword_list_str = ", ".join(keywords)

    # System prompt provides all instructions
    system_prompt = f"""
    You are a concise legal dictionary assistant. You will be given a list of terms.
    Your task is to provide a concise, one-sentence definition (max 15 words) for each term.

    RULES:
    1.  For actual *legal or technical* terms (e.g., "liability", "jurisdiction"), provide a definition.
    2.  For simple/common English words (e.g., "and", "the", "document"), you MUST return the exact string "No explanation needed" as the value.
    3.  You MUST return your response as a single, valid JSON object, where each
        keyword is a key and its definition (or "No explanation needed") is the value.

    Example:
    {{"liability": "A legal responsibility for one's actions or debts.", "contract": "A legally binding agreement.", "with": "No explanation needed"}}
    """
    
    # User prompt provides only the data
    user_prompt = f"Here is the list of keywords: {keyword_list_str}"

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Recommended model for JSON tasks
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024, # Give it enough space for definitions
            response_format={"type": "json_object"}  # <-- 1. GUARANTEES JSON OUTPUT
        )

        # Extract the response content (which is now a guaranteed JSON string)
        result_text = response.choices[0].message.content
        
        # --- 2. RELIABLE JSON PARSING ---
        try:
            # Safely parse the guaranteed JSON string into a Python dictionary
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            # This should rarely happen, but it's good practice to have it
            return {"error": f"Failed to decode the JSON response from API: {e}"}
        # --- End of fix ---

    except Exception as e:
        return {"error": f"⚠️ Error fetching meanings: {e}"}