# Social Media Language Analyser

A modular Python pipeline for sourcing, cleaning, rewriting, and analyzing text data from Reddit and Bluesky to create structured corpora for NLP research.

## Description

This project provides a comprehensive, end-to-end solution for building high-quality text corpora from social media platforms. It features a robust, modular architecture that separates concerns into distinct components for data sourcing, cleaning, LLM-powered rewriting, and linguistic analysis. The system includes four independent pipelines to collect both original and LLM-rewritten posts from Reddit and Bluesky, ensuring data integrity and variety. A key focus of the project is compliance with platform Terms of Service; it is designed for non-training NLP tasks such as model evaluation, RAG, and linguistic research. The final output is a set of well-structured, clean JSONL files, complete with rich metadata for full traceability, and a separate analysis pipeline to evaluate the linguistic characteristics of the generated corpora.

## Getting Started

### Dependencies

* **Python 3.9+**
* All required Python packages are listed in the `requirements.txt` file.
* The scripts are OS-agnostic but have been tested on macOS and Linux. Windows users may need to adjust path separators if issues arise.

### Installing

1.  **Clone the repository** to your local machine:
    ```bash
    git clone https://github.com/mr-gribbles/sm-language-analyser.git
    cd sm-language-analyser
    ```
2.  **Set up a Python virtual environment** to isolate dependencies:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install all required packages**:
    ```bash
    pip install -r requirements.txt
    ```

### Executing program

1.  **Run the interactive setup script** to create your `.env` file with the necessary API keys. The script will prompt you for each required credential.
    ```bash
    python setup_env.py
    ```
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
    ```bash
    python analyze_corpus.py path/to/your/corpus_file.jsonl
    ```

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

* **mr-gribbles**
* [@mr-gribbles](https://github.com/mr-gribbles)

## Version History

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


