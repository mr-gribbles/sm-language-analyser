"""Text generation using the Gemini API.

This module provides functionality to generate text using Google's Gemini API
with robust error handling for blocked prompts and API errors. The generation
function takes a prompt and generates new text using a specified Gemini model.
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


def process_text_with_gemini(prompt, model_name):
    """Use the Gemini API to process text from the provided prompt.

    Includes robust handling for blocked prompts and API errors.

    Args:
        prompt: The prompt to use for text generation.
        model_name: The name of the Gemini model to use for generation.

    Returns:
        str or None: The processed text if successful, otherwise None. Returns None if
            the prompt is blocked by safety filters or if an API error occurs.
    """
    try:
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(prompt)

        # If the response has no 'candidates', it means the prompt was blocked
        # by the safety filters.
        if not response.candidates:
            print("Warning: A prompt was blocked by the safety filter. Skipping.")
            return None

        return response.text.strip()

    except Exception as e:
        # This will catch other API errors, like connection issues.
        print(f"An error occurred with the Gemini API: {e}")
        return None
