import os
import google.generativeai as genai
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded before they are used.
load_dotenv()

# Load the API key from the .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the API client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")

def rewrite_text_with_gemini(text_to_rewrite, model_name, prompt_template):
    """
    Uses the Gemini API to rewrite the provided text, now with robust
    handling for blocked prompts.

    Args:
        text_to_rewrite: The cleaned text from the social media post.
        model_name: The specific Gemini model to use.
        prompt_template: The instructional string with a placeholder for the text.

    Returns:
        The rewritten text as a string, or None if an error or block occurs.
    """
    try:
        model = genai.GenerativeModel(model_name)
        prompt = prompt_template.format(text_to_rewrite=text_to_rewrite)
        
        response = model.generate_content(prompt)
        
        # If the response has no 'candidates', it means the prompt was blocked
        # by the safety filters.
        if not response.candidates:
            # You can optionally inspect the reason for the block
            # print(f"Warning: Prompt was blocked. Reason: {response.prompt_feedback.block_reason}")
            print("Warning: A post was blocked by the safety filter. Skipping.")
            return None

        # If the prompt was not blocked, we can safely access the text.
        return response.text.strip()
        
    except Exception as e:
        # This will catch other API errors, like connection issues.
        print(f"An error occurred with the Gemini API: {e}")
        return None
