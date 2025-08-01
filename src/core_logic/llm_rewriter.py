"""Text rewriting using the Gemini API.

This module provides functionality to rewrite text using Google's Gemini API
with robust error handling for blocked prompts and API errors. The rewrite
function takes cleaned text and transforms it using a specified Gemini model
and prompt template.
"""
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
    raise ValueError(
        "Gemini API key not found. Please set GEMINI_API_KEY in your .env file."
    )


def rewrite_text_with_gemini(text_to_rewrite, model_name, prompt_template):
    """Use the Gemini API to rewrite the provided text.

    Includes robust handling for blocked prompts and API errors.

    Args:
        text_to_rewrite: The cleaned text that needs to be rewritten.
        model_name: The name of the Gemini model to use for rewriting.
        prompt_template: The template for the prompt to be used with the model.

    Returns:
        The rewritten text if successful, otherwise None. Returns None if
        the prompt is blocked by safety filters or if an API error occurs.
    """
    try:
        model = genai.GenerativeModel(model_name)
        prompt = prompt_template.format(text_to_rewrite=text_to_rewrite)
        
        response = model.generate_content(prompt)
        
        # If the response has no 'candidates', it means the prompt was blocked by the safety filters.
        if not response.candidates:
            print("Warning: A post was blocked by the safety filter. Skipping.")
            return None
        return response.text.strip()
        
    except Exception as e:
        # This will catch other API errors, like connection issues.
        print(f"An error occurred with the Gemini API: {e}")
        return None
