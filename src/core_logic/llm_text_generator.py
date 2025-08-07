"""Text generation using the Gemini API.

This module provides functionality to generate text using Google's Gemini API
with robust error handling for blocked prompts and API errors. The generation
function takes prompts and transforms them using a specified Gemini model.
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


def generate_text_with_gemini(prompt, model_name):
    """Use the Gemini API to generate text from the provided prompt.

    Includes robust handling for blocked prompts and API errors.

    Args:
        prompt: The prompt to send to the model for text generation.
        model_name: The name of the Gemini model to use for generation.

    Returns:
        The generated text if successful, otherwise None. Returns None if
        the prompt is blocked by safety filters or if an API error occurs.
    """
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # If the response has no 'candidates', it means the prompt was blocked by the safety filters.
        if not response.candidates:
            return None
        return response.text.strip()
        
    except Exception:
        # This will catch other API errors, like connection issues.
        return None
