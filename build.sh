#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Install Python dependencies
pip install -r requirements.txt

# Create NLTK data directory and download data
mkdir -p /opt/venv/nltk_data
python -m nltk.downloader -d /opt/venv/nltk_data vader_lexicon punkt punkt_tab

echo "Build completed successfully!"
