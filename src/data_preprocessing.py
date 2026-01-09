import re

def preprocess_code(text, is_code=True):
    if text is None or text == "": 
        return ""
    
    if is_code:
        # Remove comments 
        text = re.sub(r'#.*', '', text)
        
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\s+', ' ', text).strip()
        
        # CamelCase splitting: calculateBLUE -> calculate BLUE
        text = re.sub('([a-z0-9])([A-Z])', r'\1 \2', text)
        
        # Snake_case splitting: calculate_blue -> calculate blue
        text = text.replace('_', ' ')
    else:
        # For summaries, just normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text.lower()