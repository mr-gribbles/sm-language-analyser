# Social Media Language Analyser

> **Note:** This branch contains the web interface version. For the most up-to-date command-line version and core features, please refer to the `v2.0-refactor` branch. New features will be rolled out to the web app version over time.

A modular Python pipeline for sourcing, cleaning, rewriting, and analyzing text data from Reddit and Bluesky to create structured corpora for NLP research.

## Description

This project provides a comprehensive, end-to-end solution for building high-quality text corpora from social media platforms. It features a robust, modular architecture that separates concerns into distinct components for data sourcing, cleaning, LLM-powered rewriting, and linguistic analysis. The system includes four independent pipelines to collect both original and LLM-rewritten posts from Reddit and Bluesky, ensuring data integrity and variety. 

**Check out my website for a publicly avaliable web client:** [https://sm-language.up.railway.app](https://sm-language.up.railway.app)

## Getting Started with the Web App

This guide will walk you through setting up and running the web interface using a Python virtual environment.

### Step 1: Initial Setup

1.  **Clone this specific branch** to your local machine:
    ```bash
    git clone https://github.com/mr-gribbles/sm-language-analyser.git --branch feature/web-gui
    cd sm-language-analyser
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download required NLTK data:**
    ```bash
    python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
    ```

### Step 2: Configuration

1.  **Create a `.env` file.** This file will store your API keys and other configuration settings. You can create it by copying the example file:
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file** and add your credentials:
    *   **`REDDIT_CLIENT_ID` & `REDDIT_CLIENT_SECRET`:** Create a new "script" application at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
    *   **`GEMINI_API_KEY`:** Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and click "Create API key".
    *   **`BLUESKY_USERNAME` & `BLUESKY_PASSWORD`:** In the Bluesky app, go to Settings > Advanced > App Passwords to generate a new password. **Do not use your main account password.**

### Step 3: Running the Web Application

1.  **Ensure your virtual environment is activated:**
    ```bash
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2.  **Start the Flask application:**
    ```bash
    python app.py
    ```

3.  **Open your browser** and navigate to [http://localhost:5001](http://localhost:5001).

### Using the Web Interface

The web application provides a modern, user-friendly interface for all the project's features:

*   **Pipeline Runner:** Navigate to the main page to run the data collection pipeline. You can select the platform (Reddit, Bluesky, or both), choose whether to rewrite posts with LLM, set collection limits, and see live logs of the process.

*   **Corpus Management:** Navigate to the "Corpus Management" page to view all your generated corpus files. From here, you can:
    - **Analyze Corpus Files:** Generate detailed linguistic analysis including readability metrics, lexical diversity, and sentiment analysis
    - **Combine Corpora:** Merge multiple corpus files into a single file
    - **Concordance Analysis:** Perform advanced comparative analysis between original and rewritten corpora, with results displayed in a new browser tab

### Features

- **Modern UI:** Beautiful, responsive interface with glass-morphism design
- **Real-time Logging:** Live streaming of all pipeline and analysis operations
- **Comprehensive Analysis:** Multiple analysis tools for linguistic research
- **File Management:** Easy corpus file organization and management
- **Cross-platform:** Works on Windows, macOS, and Linux

## Running Tests

This project uses `pytest` for unit testing. To run the tests, ensure your virtual environment is activated and run:

```bash
pytest
```

## Troubleshooting

### Common Issues

1. **SSL Certificate Error (especially on macOS):**
   If you encounter `[SSL: CERTIFICATE_VERIFY_FAILED]` errors when downloading NLTK data:
   
   **Option 1:** Run the Python Certificate Installer:
   * Navigate to your Python installation folder (usually in `/Applications/Python 3.11/`).
   * Double-click on `Install Certificates.command`.

   **Option 2:** Manually download NLTK data:
   ```bash
   python -m nltk.downloader vader_lexicon punkt punkt_tab
   ```

2. **Port Already in Use:**
   If port 5001 is already in use, you can kill the process:
   ```bash
   lsof -ti:5001 | xargs kill -9
   ```

3. **Missing Dependencies:**
   If you encounter import errors, ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   pip install pyarrow textstat nltk pandas
   ```

### Virtual Environment Management

- **Activate environment:** `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
- **Deactivate environment:** `deactivate`
- **Recreate environment:** Delete `.venv` folder and repeat setup steps

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
