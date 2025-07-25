# llm_rewriter.py
import os
import google.generativeai as genai

# Load the API key from the .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    # This check prevents the app from running without credentials.
    raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")

def rewrite_text_with_gemini(text_to_rewrite, model_name, prompt_template):
    """
    Uses the Gemini API to rewrite the provided text based on a prompt template.

    Args:
        text_to_rewrite: The cleaned text from the Reddit post.
        model_name: The specific Gemini model to use (e.g., 'gemini-1.5-flash-latest').
        prompt_template: The instructional string with a placeholder for the text.

    Returns:
        The rewritten text as a string, or None if an error occurs.
    """
    try:
        model = genai.GenerativeModel(model_name)
        # Format the prompt with the actual text
        prompt = prompt_template.format(text_to_rewrite=text_to_rewrite)
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return None
