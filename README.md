# Social Media Language Analyser

> **Note:** This branch is for the railway web interface version. For the most up-to-date command-line version and core features, please refer to the `main` branch. New features will be rolled out to the web app version over time. Otherwise , check out the `feature/web-api` to run a local version of the site. 

A modular Python pipeline for sourcing, cleaning, rewriting, and analyzing text data from Reddit and Bluesky to create structured corpora for NLP research.

## Description

This project provides a comprehensive, end-to-end solution for building high-quality text corpora from social media platforms. It features a robust, modular architecture that separates concerns into distinct components for data sourcing, cleaning, LLM-powered rewriting, and linguistic analysis. The system includes four independent pipelines to collect both original and LLM-rewritten posts from Reddit and Bluesky, ensuring data integrity and variety. A key focus of the project is compliance with platform Terms of Service; it is designed for non-training NLP tasks such as model evaluation, RAG, and linguistic research. The final output is a set of well-structured, clean JSONL files, complete with rich metadata for full traceability, and a separate analysis pipeline to evaluate the linguistic characteristics of the generated corpora.

**Check out my website for a publicly avaliable web client:** [https://sm-language.up.railway.app](https://sm-language.up.railway.app)

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
