import re

def clean_text(text):
    """
    Cleans text by:
    1. Lowercasing
    2. Removing non-alphabetic/non-numeric characters (punctuation, special chars)
    3. Normalizing whitespace
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove non-alphanumeric except spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
