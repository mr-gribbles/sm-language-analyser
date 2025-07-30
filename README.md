# Social Media Language Analyser

> **Note:** This branch contains the Dockerized web interface. For the most up-to-date command-line version and core features, please refer to the `v2.0-refactor` branch. New features will be rolled out to the web app version over time.

A modular Python pipeline for sourcing, cleaning, rewriting, and analyzing text data from Reddit and Bluesky to create structured corpora for NLP research.

## Description

This project provides a comprehensive, end-to-end solution for building high-quality text corpora from social media platforms. It features a robust, modular architecture that separates concerns into distinct components for data sourcing, cleaning, LLM-powered rewriting, and linguistic analysis. The system includes four independent pipelines to collect both original and LLM-rewritten posts from Reddit and Bluesky, ensuring data integrity and variety. A key focus of the project is compliance with platform Terms of Service; it is designed for non-training NLP tasks such as model evaluation, RAG, and linguistic research. The final output is a set of well-structured, clean JSONL files, complete with rich metadata for full traceability, and a separate analysis pipeline to evaluate the linguistic characteristics of the generated corpora.

## Getting Started with the Web App

This guide will walk you through setting up and running the Dockerized web interface.

### Step 1: Initial Setup

1.  **Clone this specific branch** to your local machine:
    ```bash
    git clone https://github.com/mr-gribbles/sm-language-analyser.git --branch feature/web-gui
    cd sm-language-analyser
    ```
2.  **Ensure you have Docker installed.** You can find installation instructions at the [official Docker website](https://docs.docker.com/get-docker/).

### Step 2: Configuration

1.  **Create a `.env` file.** This file will store your API keys and other configuration settings. You can create it by copying the example file:
    ```bash
    cp .env.example .env
    ```
2.  **Edit the `.env` file** and add your credentials.
    *   **`REDDIT_CLIENT_ID` & `REDDIT_CLIENT_SECRET`:** Create a new "script" application at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
    *   **`GEMINI_API_KEY`:** Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and click "Create API key".
    *   **`BLUESKY_USERNAME` & `BLUESKY_PASSWORD`:** In the Bluesky app, go to Settings > Advanced > App Passwords to generate a new password. **Do not use your main account password.**

### Step 3: Running the Web Application

1.  **Build and run the application using Docker Compose:**
    ```bash
    docker-compose up --build
    ```
2.  **Open your browser** and navigate to [http://localhost:5001](http://localhost:5001).

### Using the Web Interface

The web application provides a user-friendly interface for all the project's features:

*   **Pipeline Runner:** Navigate to the main page to run the data collection pipeline. You can select the platform, choose whether to rewrite posts, and see the live logs of the process.
*   **Corpus Management:** Navigate to the "Corpus Management" page to view all your generated corpus files. From here, you can trigger the analysis, combination, and concordance scripts, with all output streamed in real-time to the logs window.

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
