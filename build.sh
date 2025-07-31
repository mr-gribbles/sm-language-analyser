#!/bin/bash
# Railway build script

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True)"

echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo "Build completed successfully!"
