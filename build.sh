#!/bin/bash
# Railway build script

echo "Installing lightweight dependencies..."
pip install -r requirements-railway.txt

echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True)"

echo "Build completed successfully!"
