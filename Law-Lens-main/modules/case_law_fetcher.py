import os
import json
import logging
from typing import Dict, Tuple
from groq import Groq

logging.basicConfig(level=logging.INFO)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_ipc_section_for_keyword(keyword: str) -> str:
    """
    Identify the most relevant IPC section or law for the keyword.
    Returns IPC section number or law reference.
    """
    prompt = f"""You are an Indian legal expert specializing in the Indian Penal Code (IPC) and other Indian laws.

For the legal keyword: "{keyword}"

Identify the MOST RELEVANT IPC section or other Indian law section that relates to this keyword.

RULES:
- If it's a criminal matter, provide IPC section (e.g., "IPC 420", "IPC 302")
- If it's civil/contract law, provide relevant act (e.g., "Contract Act Section 73", "Sale of Goods Act Section 55")
- If it's property law, provide relevant act (e.g., "Transfer of Property Act Section 54")
- If no specific section applies, return "General Legal Term"

Respond with ONLY the section reference, nothing else.
Examples: "IPC 420", "IPC 406", "Contract Act Section 73", "General Legal Term"

Your response:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        
        ipc_section = response.choices[0].message.content.strip()
        logging.info(f"IPC section for '{keyword}': {ipc_section}")
        return ipc_section
        
    except Exception as e:
        logging.error(f"Error getting IPC section for '{keyword}': {e}")
        return "General Legal Term"


def get_case_law_for_keyword(keyword: str) -> Tuple[Dict, str]:
    """
    Identify IPC section for keyword and create generic IndianKanoon link.
    Shows ALL cases related to that IPC section.
    
    Returns:
        Tuple of (json_output, ui_output)
    """
    
    # Get IPC section for keyword
    ipc_section = get_ipc_section_for_keyword(keyword)
    
    # Determine category based on IPC section
    if "IPC" in ipc_section:
        category = "criminal"
    elif "Contract Act" in ipc_section:
        category = "civil"
    elif "Consumer" in ipc_section:
        category = "consumer"
    elif "Property" in ipc_section or "Transfer" in ipc_section:
        category = "property"
    else:
        category = "general"
    
    # Create generic IndianKanoon search link for the IPC section
    # This will show ALL cases related to this IPC section
    search_query = ipc_section.replace(" ", "+")
    kanoon_link = f"https://indiankanoon.org/search/?formInput={search_query}"
    
    # Create summary based on IPC section
    summary = f"Click the link below to view all Indian court cases involving {ipc_section}. This will show Supreme Court and High Court judgments where this law section was central to the dispute."
    
    # Build JSON output
    json_output = {
        "keyword": keyword,
        "ipc_section": ipc_section,
        "case_category": category,
        "summary": summary,
        "search_query": search_query,
        "kanoon_link": kanoon_link
    }
    
    # Build UI output
    ui_output = f"""Keyword: {keyword}
IPC/Law Section: {ipc_section}
Category: {category.title()}
Summary: {summary}
IndianKanoon Link: {kanoon_link}"""
    
    logging.info(f"Generated case law link for '{keyword}' → {ipc_section} → {kanoon_link}")
    
    return json_output, ui_output


def get_cases_for_keywords(keywords: list) -> Dict[str, Tuple[Dict, str]]:
    """
    Fetch case law links for multiple keywords.
    
    Returns:
        Dict mapping keyword -> (json_output, ui_output)
    """
    if not keywords:
        logging.warning("No keywords provided for case law fetching")
        return {}
    
    results = {}
    
    logging.info(f"Starting case law fetch for {len(keywords)} keywords: {keywords}")
    
    for i, keyword in enumerate(keywords, 1):
        logging.info(f"Processing keyword {i}/{len(keywords)}: {keyword}")
        
        try:
            json_output, ui_output = get_case_law_for_keyword(keyword)
            results[keyword] = (json_output, ui_output)
            logging.info(f"Successfully processed '{keyword}'")
        except Exception as e:
            logging.error(f"Failed to process '{keyword}': {e}")
            results[keyword] = (
                {"keyword": keyword, "error": str(e)},
                f"Keyword: {keyword}\nError occurred while processing."
            )
        
        # Small delay to avoid rate limits
        if i < len(keywords):
            import time
            time.sleep(0.5)
    
    logging.info(f"Case law processing complete. Retrieved {len(results)} results.")
    return results
