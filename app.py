import os
import sys
import time
import glob
from flask import Flask, render_template, request, Response, jsonify
from threading import Thread

# Add the project root to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import run_pipeline
from scripts.analyze_corpus import analyze_corpus_file
from scripts.combine_corpora import combine_jsonl_files
from scripts.conc_analysis import run_conc_analysis

app = Flask(__name__)

# A simple in-memory log store
logs = []

def stream_logs():
    """A generator function to stream logs."""
    # A simple way to stream logs: yield them as they are added
    # In a real-world app, you might use a more robust queue like Redis
    last_yielded_index = 0
    while True:
        if len(logs) > last_yielded_index:
            for i in range(last_yielded_index, len(logs)):
                yield f"data: {logs[i]}\n\n"
            last_yielded_index = len(logs)
        time.sleep(0.1) # prevent busy-waiting

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/corpus')
def corpus_management():
    """Renders the corpus management page."""
    return render_template('corpus.html')

@app.route('/api/corpus_files')
def get_corpus_files():
    """Returns a list of all .jsonl files in the corpora directory."""
    original_files = glob.glob('corpora/original_only/*.jsonl')
    rewritten_files = glob.glob('corpora/rewritten_pairs/*.jsonl')
    return jsonify({
        'original': [os.path.basename(f) for f in original_files],
        'rewritten': [os.path.basename(f) for f in rewritten_files]
    })

@app.route('/run', methods=['POST'])
def run():
    """Runs the data collection pipeline."""
    platform = request.form.get('platform')
    rewrite = 'rewrite' in request.form
    num_posts = int(request.form.get('num_posts', 100))
    reddit_limit = int(request.form.get('reddit_limit', 300))
    bluesky_limit = int(request.form.get('bluesky_limit', 100))
    
    # Clear previous logs
    logs.clear()

    # Run the pipeline in a separate thread to avoid blocking the web server
    thread = Thread(target=run_pipeline_with_logging, args=(platform, rewrite, num_posts, reddit_limit, bluesky_limit))
    thread.start()
    
    return "Pipeline started! Check the logs for progress.", 200

@app.route('/logs')
def log_stream():
    """Streams the pipeline logs to the client."""
    return Response(stream_logs(), mimetype='text/event-stream')

def run_script_with_logging(target_func, *args):
    """
    A wrapper to run a function and capture its print statements.
    """
    # Redirect stdout to capture logs
    class LogCatcher:
        def write(self, message):
            if message.strip():
                # Split multi-line messages and add each line separately
                lines = message.strip().split('\n')
                for line in lines:
                    if line.strip():
                        logs.append(line.strip())
        
        def flush(self):
            pass

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = LogCatcher()
    sys.stderr = LogCatcher()
    
    try:
        target_func(*args)
        logs.append("--- SCRIPT FINISHED ---")
    except BaseException as e:
        import traceback
        error_details = traceback.format_exc().replace('\n', '<br>')
        logs.append(f"<div style='color: red;'>--- SCRIPT FAILED: {e}<br>{error_details} ---</div>")
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def run_pipeline_with_logging(platform, rewrite, num_posts, reddit_limit, bluesky_limit):
    run_script_with_logging(run_pipeline, platform, rewrite, num_posts, reddit_limit, bluesky_limit)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Runs the corpus analysis script."""
    filepath = request.form.get('filepath')
    logs.clear()
    thread = Thread(target=run_script_with_logging, args=(analyze_corpus_file, filepath))
    thread.start()
    return "Analysis started!", 200

@app.route('/combine', methods=['POST'])
def combine():
    """Runs the corpus combination script."""
    directory = request.form.get('directory')
    delete_originals = 'delete_originals' in request.form
    logs.clear()
    thread = Thread(target=run_script_with_logging, args=(combine_jsonl_files, directory, delete_originals))
    thread.start()
    return "Combination started!", 200

@app.route('/conc_analyze', methods=['POST'])
def conc_analyze():
    """Runs the concordance analysis script."""
    original_corpus = request.form.get('original_corpus')
    rewritten_corpus = request.form.get('rewritten_corpus')
    logs.clear()
    thread = Thread(target=run_script_with_logging, args=(run_conc_analysis, original_corpus, rewritten_corpus))
    thread.start()
    return "Concordance analysis started!", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
