# Social Media Language Analyser

A modular Python pipeline for sourcing, cleaning, rewriting, and analyzing text data from Reddit and Bluesky to create structured corpora for NLP research.

## Description

This project provides a comprehensive, end-to-end solution for building high-quality text corpora from social media platforms. It features a robust, modular architecture that separates concerns into distinct components for data sourcing, cleaning, LLM-powered rewriting, and linguistic analysis. The system includes four independent pipelines to collect both original and LLM-rewritten posts from Reddit and Bluesky, ensuring data integrity and variety. A key focus of the project is compliance with platform Terms of Service; it is designed for non-training NLP tasks such as model evaluation, RAG, and linguistic research. The final output is a set of well-structured, clean JSONL files, complete with rich metadata for full traceability, and a separate analysis pipeline to evaluate the linguistic characteristics of the generated corpora.

## Usage Workflow

This guide will walk you through the entire process of setting up the project, collecting data, and analyzing it.

### Step 1: Initial Setup

1.  **Clone the repository** to your local machine:
    ```bash
    git clone https://github.com/mr-gribbles/sm-language-analyser.git
    cd sm-language-analyser
    ```
2.  **Set up a Python virtual environment** to isolate dependencies. This is a crucial step to avoid conflicts with other projects.
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Generate and install the required packages.** The `generate_requirements.py` script scans the project for all imported packages and creates a `requirements.txt` file.
    ```bash
    python scripts/generate_requirements.py
    pip install -r requirements.txt
    ```

### Step 2: Configuration

1.  **Create a `.env` file.** This file will store your API keys and other configuration settings. You can create it by copying the example file:
    ```bash
    cp .env.example .env
    ```
2.  **Edit the `.env` file** and add your credentials.
    *   **`REDDIT_CLIENT_ID` & `REDDIT_CLIENT_SECRET`:** Create a new "script" application at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
    *   **`GEMINI_API_KEY`:** Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and click "Create API key".
    *   **`BLUESKY_USERNAME` & `BLUESKY_PASSWORD`:** In the Bluesky app, go to Settings > Advanced > App Passwords to generate a new password. **Do not use your main account password.**

### Step 3: Data Collection

Run the main pipeline using `main.py`, specifying the platform (`reddit` or `bluesky`) and whether you want to rewrite the posts.

*   **Collect original posts from Reddit:**
    ```bash
    python main.py reddit
    ```
*   **Collect and rewrite posts from Reddit:**
    ```bash
    python main.py reddit --rewrite
    ```
*   **Collect original posts from Bluesky:**
    ```bash
    python main.py bluesky
    ```
*   **Collect and rewrite posts from Bluesky:**
    ```bash
    python main.py bluesky --rewrite
    ```
Collected data will be saved as `.jsonl` files in the `corpora/original_only` or `corpora/rewritten_pairs` directories.

### Step 4: Corpus Analysis

Once you have generated a corpus file, you can analyze its linguistic features using the `analyze_corpus.py` script.

*   **Analyze a corpus file:**
    ```bash
    python scripts/analyze_corpus.py corpora/original_only/your_file.jsonl
    ```
    Replace `your_file.jsonl` with the name of the file you want to analyze. The script will generate a `_analysis.csv` file with the results.

### Step 5: Combining Corpora (Optional)

If you have multiple `.jsonl` files, you can merge them into a single file for easier analysis.

*   **Combine all original corpus files:**
    ```bash
    python scripts/combine_corpora.py corpora/original_only
    ```
*   **To also delete the original files after merging, add the `--delete-originals` flag:**
    ```bash
    python scripts/combine_corpora.py corpora/original_only --delete-originals
    ```

## Running Tests

This project uses `pytest` for unit testing. To run the tests, simply run the following command from the root directory:

```bash
pytest
```

## Help

A common issue, especially on macOS, is an `[SSL: CERTIFICATE_VERIFY_FAILED]` error when the analysis script tries to download NLTK data. If this happens, you have two options:

1.  **Run the Python Certificate Installer:**
    * Navigate to your Python installation folder (usually in `/Applications/Python 3.11/`).
    * Double-click on `Install Certificates.command`.

2.  **Manually download the NLTK data** from your terminal (ensure your virtual environment is active):
    ```bash
    python -m nltk.downloader vader_lexicon punkt punkt_tab
    ```

## Authors

* **Logan Barrington**
* [@mr-gribbles](https://github.com/mr-gribbles)

## Version History

* **2.0**
    * **Complete Codebase Refactor:** Overhauled the project structure for modularity and maintainability.
    * **Unified Pipeline:** Replaced four separate main scripts with a single `main.py` entry point using command-line arguments.
    * **Centralized Logic:** Moved all core data processing logic into a new `src/pipeline.py` module.
    * **Improved Configuration:** Migrated all settings to a `.env` file, removing hardcoded values.
    * **Standardized Naming:** Refactored function and variable names for clarity and consistency.
    * **Added Unit Tests:** Integrated `pytest` and added an initial test suite for core logic.
    * **Improved Directory Structure:** Created a `scripts` directory for utility scripts and a `tests` directory for all tests.
* **1.3**
    * Added linguistic analysis pipeline (`analyze_corpus.py`).
    * Added Bluesky scraping and rewriting capabilities.
    * Restructured project into a modular `src` directory.
* **1.2**
    * Implemented modular Reddit pipelines for original and rewritten corpora.
    * Added `config.py` for centralized settings.
    * Integrated Google Gemini for LLM rewriting tasks.
* **1.1**
    * Initial Release: Basic Reddit scraping and PRAW setup.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
