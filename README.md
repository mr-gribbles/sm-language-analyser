# Social Media Language Analyser

> **Branch Overview:** This branch contains the modern, command-line version of the application. It is the primary development branch for core features. For the legacy version, see the `legacy/v1.0` branch. For the web interface, see the `feature/web-gui` branch.

A modular Python pipeline for sourcing, cleaning, rewriting, and analyzing text data from Reddit and Bluesky to create structured corpora for NLP research.

## Description

This project provides a comprehensive, end-to-end solution for building high-quality text corpora from social media platforms. It features a robust, modular architecture that separates concerns into distinct components for data sourcing, cleaning, LLM-powered rewriting, and linguistic analysis. The system includes four independent pipelines to collect both original and LLM-rewritten posts from Reddit and Bluesky, ensuring data integrity and variety. 

**Check out my website for a publicly avaliable web client:** [https://sm-language.up.railway.app](https://sm-language.up.railway.app)

## Usage Workflow

This guide will walk you through the entire process of setting up the project, collecting data, and analyzing it.

### Step 1: Initial Setup

1.  **Clone this specific branch** to your local machine:
    ```bash
    git clone https://github.com/mr-gribbles/sm-language-analyser.git --branch v2.0-refactor
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

### Step 6: Advanced Analysis

The `conc_analysis.py` script performs a keyness analysis to compare the original and rewritten corpora. The required spaCy model (`en_core_web_sm`) will be downloaded automatically the first time you run the script.

*   **Run the advanced analysis:**
    ```bash
    python scripts/conc_analysis.py corpora/original_only/your_file.jsonl corpora/rewritten_pairs/your_rewritten_file.jsonl
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



* **2.2**
    * **Modernized Web Interface:** Complete UI/UX overhaul with modern design, responsive layout, and improved accessibility.
    * **Removed Docker Dependency:** Simplified deployment by using Python virtual environments instead of Docker containers.
    * **Enhanced Analysis Tools:** Improved concordance analysis with HTML report generation and automatic browser opening.
    * **Fixed Text Overflow Issues:** Resolved UI problems with long filenames and improved responsive design.
    * **Better Error Handling:** Enhanced error messages and troubleshooting guidance.
* **2.1**
    * **Added Web Interface:** Created a Flask-based web GUI for running the pipeline and managing corpora.
    * **Dockerized Application:** Added containerization for easy deployment (later removed in v2.2).
    * **Integrated Analysis Tools:** The web interface includes controls for running analysis, combine, and concordance scripts.
    * **Real-Time Logging:** Implemented real-time log streaming to the web interface.
    * **Automated SpaCy Model Download:** Scripts now automatically download required models.
* **2.0**
    * **Complete Codebase Refactor:** Overhauled project structure for modularity and maintainability.
    * **Unified Pipeline:** Replaced four separate scripts with a single entry point.
    * **Centralized Logic:** Moved core processing logic into modular components.
    * **Improved Configuration:** Migrated all settings to `.env` file.
    * **Added Unit Tests:** Integrated `pytest` with comprehensive test suite.
* **1.3**
    * Added linguistic analysis pipeline and Bluesky integration.
    * Restructured project into modular `src` directory.
* **1.2**
    * Implemented modular Reddit pipelines and Google Gemini integration.
* **1.1**
    * Initial Release: Basic Reddit scraping functionality.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
