# Social Media Language Analyser

> **⚠️ This is a legacy version of the project and is no longer maintained. Please refer to the `v2.0-refactor` branch for the latest version.**

A modular Python pipeline for sourcing, cleaning, rewriting, and analyzing text data from Reddit and Bluesky to create structured corpora for NLP research.

## Description

This project provides a comprehensive, end-to-end solution for building high-quality text corpora from social media platforms. It features a robust, modular architecture that separates concerns into distinct components for data sourcing, cleaning, LLM-powered rewriting, and linguistic analysis. The system includes four independent pipelines to collect both original and LLM-rewritten posts from Reddit and Bluesky, ensuring data integrity and variety. A key focus of the project is compliance with platform Terms of Service; it is designed for non-training NLP tasks such as model evaluation, RAG, and linguistic research. The final output is a set of well-structured, clean JSONL files, complete with rich metadata for full traceability, and a separate analysis pipeline to evaluate the linguistic characteristics of the generated corpora.

## Getting Started

### Dependencies

* **Python 3.11+**
* All required Python packages are listed in the `requirements.txt` file.
* The scripts are OS-agnostic but have been tested on macOS and Linux. Windows users may need to adjust path separators if issues arise.

### Installing

1.  **Clone this specific branch** to your local machine:
    ```bash
    git clone https://github.com/mr-gribbles/sm-language-analyser.git --branch legacy/v1.0
    cd sm-language-analyser
    ```
2.  **Set up a Python virtual environment** to isolate dependencies:
    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Generate then install all required packages**:
    ```bash
    python generate_requirments.py
    pip install -r requirements.txt
    ```

### Executing program

1.  **Run the interactive setup script** to create your `.env` file with the necessary API keys. The script will prompt you for each required credential.
    ```bash
    python setup_env.py
    ```
     #### Where to Find Your API Keys
    * **Reddit Client ID & Secret:** Log in to Reddit, then create a new "script" application at [**reddit.com/prefs/apps**](https://www.reddit.com/prefs/apps).
    * **Google Gemini API Key:** Go to [**Google AI Studio**](https://aistudio.google.com/app/apikey) and click "Create API key".
    * **Bluesky App Password:** In the Bluesky app, go to Settings > Advanced > App Passwords to generate a new password. **Do not use your main account password.**

2.  **Choose a pipeline to run.** Based on the data you want to collect, run one of the following main scripts from your terminal:

    * To collect **original, untouched posts from Reddit**:
        ```bash
        python main_reddit_unedited.py
        ```
    * To collect and **rewrite posts from Reddit** using the LLM:
        ```bash
        python main_reddit_rewriter.py
        ```
    * To collect **original, untouched posts from Bluesky**:
        ```bash
        python main_bluesky_unedited.py
        ```
    * To collect and **rewrite posts from Bluesky** using the LLM:
        ```bash
        python main_bluesky_rewriter.py
        ```
3.  **Analyze a generated corpus file.** Once a `.jsonl` file has been created, you can run the analysis pipeline on it:
    * To analyse an **original** corpus file:
        ```bash
        python analyze_corpus.py corpora/original_only/corpus_file.jsonl
        ```
    * To analyse a **rewritten** corpus file:
        ```bash
        python analyze_corpus.py corpora/rewritten_pairs/corpus_file.jsonl
        ```
    Replace `corpus_file.jsonl` with the name of a generated corpus file.
    
4.  **Combine Corpus Files (Optional).** After running the collection pipelines multiple times, you can combine the smaller `.jsonl` files into a single, larger file for easier analysis. This script will also delete the original files after a successful merge.

    * To combine all **original** corpus files:
        ```bash
        python combine_corpora.py corpora/original_only
        ```
    * To combine all **rewritten** corpus files:
        ```bash
        python combine_corpora.py corpora/rewritten_pairs
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

* **2.1**
    * **Added Web Interface:** Created a Flask-based web GUI for running the pipeline and managing corpora.
    * **Dockerized Application:** Added a `Dockerfile` and `docker-compose.yml` to containerize the application for easy deployment.
    * **Integrated Analysis Tools:** The web interface now includes controls for running the `analyze`, `combine`, and `concordance` analysis scripts.
    * **Real-Time Logging:** Implemented real-time log streaming to the web interface for all pipeline and analysis operations.
    * **Automated SpaCy Model Download:** The `conc_analysis.py` script and the Docker image now automatically download the required spaCy model.
* **2.0**
    * **Complete Codebase Refactor:** Overhauled the project structure for modularity and maintainability.
    * **Unified Pipeline:** Replaced four separate main scripts with a single `main.py` entry point using command-line arguments.
    * **Centralized Logic:** Moved all core data processing logic into a new `src/pipeline.py` module.
    * **Improved Configuration:** Migrated all settings to a `.env` file, removing hardcoded values.
    * **Standardized Naming:** Refactored function and variable names for clarity and consistency.
    * **Added Unit Tests:** Integrated `pytest` and added an initial test suite for core logic.
    * **Improved Directory Structure:** Created a `scripts` directory for utility scripts and a `tests` directory for all tests.
* **0.3**
    * Added linguistic analysis pipeline (`analyze_corpus.py`).
    * Added Bluesky scraping and rewriting capabilities.
    * Restructured project into a modular `src` directory.
* **0.2**
    * Implemented modular Reddit pipelines for original and rewritten corpora.
    * Added `config.py` for centralized settings.
    * Integrated Google Gemini for LLM rewriting tasks.
* **0.1**
    * Initial Release: Basic Reddit scraping and PRAW setup.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
