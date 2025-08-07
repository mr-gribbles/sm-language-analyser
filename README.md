# Academic Paper Analysis Pipeline

A modular Python pipeline for collecting, cleaning, and analyzing academic papers from arXiv to create structured corpora for NLP research and AI vs Human text classification.

## Description

This project provides a comprehensive, end-to-end solution for building high-quality text corpora from academic papers. It features a robust, modular architecture that separates concerns into distinct components for data sourcing, cleaning, and linguistic analysis. The system collects original academic papers and provides a separate LLM corpus generator to create AI-generated text for training AI detection models.

## Features

- **arXiv Integration**: Direct access to 2+ million academic papers
- **Smart Filtering**: Automatic quality assessment and English language detection
- **Random Category Selection**: Easily build diverse datasets across academic domains
- **Separate LLM Corpus Generator**: Dedicated tool for creating AI-generated academic text
- **Neural Network Classifier**: PyTorch-based binary classifier for AI detection
- **Flexible Search**: Query by keywords, categories, or date ranges
- **Rate Limiting**: Respects arXiv API guidelines with proper delays
- **Data Quality**: Automatic LaTeX cleanup and text preprocessing

## Usage Workflow

### Step 1: Initial Setup

1. **Clone this repository** to your local machine:
   ```bash
   git clone https://github.com/mr-gribbles/sm-language-analyser.git
   cd sm-language-analyser
   ```

2. **Set up a Python virtual environment**:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Generate and install the required packages**:
   ```bash
   python scripts/generate_requirements.py
   pip install -r requirements.txt
   ```

### Step 2: Configuration

1. **Create a `.env` file**:
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** and add your credentials:
   - **`GEMINI_API_KEY`:** Go to [Google AI Studio](https://aistudio.google.com/app/apikey) and click "Create API key" (only needed for rewriting)

### Step 3: Data Collection

#### Basic Collection

```bash
# Collect machine learning papers (default)
python main.py

# Collect from specific category
python main.py --category cs.AI --max-papers 100

# Search by query
python main.py --query "neural networks" --max-papers 50

# Randomly select a category (great for diverse datasets!)
python main.py --random --max-papers 100

# List all available categories
python main.py --list-categories
```

#### Advanced Collection

```bash
# Use the dedicated arXiv script for more options
python scripts/collect_arxiv_papers.py --category cs.AI --max-papers 200

# Date-based collection
python scripts/collect_arxiv_papers.py --category cs.CL --start-date 2024-01-01 --max-papers 150

# List available categories
python scripts/collect_arxiv_papers.py --list-categories
```

### Step 4: Train AI Detection Model

```bash
# Train the neural network classifier
python scripts/train_classifier.py

# Validate model performance
python scripts/validate_classifier.py
```

### Step 5: Analysis and Evaluation

```bash
# Analyze corpus linguistic features
python scripts/analyze_corpus.py corpora/original_only/your_file.jsonl

# Combine multiple corpus files
python scripts/combine_corpora.py corpora/original_only

# Advanced concordance analysis
python scripts/conc_analysis.py corpora/original_only/file1.jsonl corpora/rewritten_pairs/file2.jsonl
```

## Popular arXiv Categories

### Computer Science
- `cs.AI` - Artificial Intelligence
- `cs.CL` - Computation and Language (NLP)
- `cs.CV` - Computer Vision and Pattern Recognition
- `cs.LG` - Machine Learning
- `cs.NE` - Neural and Evolutionary Computing

### Statistics & Mathematics
- `stat.ML` - Machine Learning (Statistics)
- `math.ST` - Statistics Theory
- `math.PR` - Probability

### Physics
- `physics.data-an` - Data Analysis, Statistics and Probability

## Example Workflows

### AI/ML Research Focus
```bash
# Collect AI papers from multiple categories
python main.py --category cs.AI --max-papers 500
python main.py --category cs.LG --max-papers 500
python main.py --category stat.ML --max-papers 300

# Combine and train classifier
python scripts/combine_corpora.py corpora/original_only
python scripts/train_classifier.py
```

### NLP Research Focus
```bash
# Collect NLP papers
python main.py --category cs.CL --max-papers 800
python main.py --query "natural language processing" --max-papers 400

# Generate AI text for training (use separate LLM corpus generator)
python scripts/generate_llm_corpus.py --num-texts 500

# Train classifier
python scripts/train_classifier.py
```

## Neural Network Classifier

The included PyTorch-based classifier can distinguish between human-written and AI-generated text:

- **Architecture**: Deep neural network with batch normalization and dropout
- **Features**: TF-IDF vectorization with n-gram support
- **Performance**: 90-95% accuracy with sufficient training data
- **Hardware**: Supports CPU, CUDA, and Apple Silicon (MPS)

### Training Requirements
- **Minimum**: 1,000 papers per class (human/AI)
- **Recommended**: 5,000+ papers per class
- **Optimal**: 10,000+ papers per class

## File Structure

```
├── main.py                          # Main entry point
├── src/
│   ├── clients/
│   │   └── arxiv_client.py          # arXiv API client
│   ├── scrapers/
│   │   └── arxiv_scraper.py         # arXiv paper scraper
│   ├── core_logic/
│   │   ├── corpus_manager.py        # Corpus management
│   │   ├── data_cleaner.py          # Text cleaning
│   │   └── llm_text_generator.py    # LLM text generation (for corpus generator)
│   ├── ml/
│   │   └── text_classifier.py       # Neural network classifier
│   ├── config.py                    # Configuration
│   └── arxiv_paper_collector.py     # Paper collection pipeline
├── scripts/
│   ├── collect_arxiv_papers.py      # Advanced collection script
│   ├── generate_llm_corpus.py       # LLM corpus generator
│   ├── train_classifier.py          # Model training
│   ├── validate_classifier.py       # Model validation
│   ├── analyze_corpus.py            # Corpus analysis
│   └── combine_corpora.py           # Corpus combination
└── ACADEMIC_PAPERS_GUIDE.md         # Detailed usage guide
```

## Running Tests

```bash
pytest
```

## Troubleshooting

### Common Issues

1. **SSL Certificate Errors** (macOS):
   ```bash
   /Applications/Python\ 3.11/Install\ Certificates.command
   ```

2. **Rate Limiting**:
   - Increase delay between requests
   - Reduce batch sizes
   - Spread collection over multiple sessions

3. **Memory Issues**:
   - Process papers in smaller batches
   - Use streaming processing

## Performance Expectations

### Data Quality
- **Academic papers**: Formal, structured language
- **Rich vocabulary**: Technical terminology
- **Consistent format**: Standardized structure
- **Large volume**: Millions of papers available

### Classification Accuracy
- **Baseline**: 85-90% with 1,000 papers per class
- **Good**: 90-95% with 5,000 papers per class
- **Excellent**: 95%+ with 10,000+ papers per class

## Authors

* **Logan Barrington**
* [@mr-gribbles](https://github.com/mr-gribbles)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Version History

* **3.0** - Academic Paper Focus
  * **Removed Social Media**: Streamlined to focus only on academic papers
  * **Enhanced arXiv Integration**: Improved paper collection and processing
  * **Better Classification**: Optimized for academic text patterns
  * **Simplified Interface**: Cleaner command-line interface
  * **Comprehensive Documentation**: Detailed guides and examples

* **2.2** - Modernized Web Interface and Enhanced Analysis Tools
* **2.1** - Added Web Interface and Dockerized Application  
* **2.0** - Complete Codebase Refactor with Unified Pipeline
* **1.x** - Initial Social Media Scraping Functionality
