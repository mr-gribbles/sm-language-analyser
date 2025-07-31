import os
import sys
import time
import glob
import json
import datetime
from flask import Flask, render_template, request, Response, jsonify, flash
from werkzeug.utils import secure_filename
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

@app.route('/reports')
def list_reports():
    """Lists all available analysis reports."""
    reports_dir = 'analysis_results'
    if not os.path.exists(reports_dir):
        return jsonify({'reports': []})
    
    reports = []
    for filename in os.listdir(reports_dir):
        if filename.endswith('.html'):
            reports.append({
                'filename': filename,
                'name': filename.replace('_concordance_report.html', '').replace('_', ' vs '),
                'url': f'/reports/{filename}'
            })
    
    return jsonify({'reports': reports})

@app.route('/reports/<filename>')
def serve_report(filename):
    """Serves analysis report files."""
    reports_dir = 'analysis_results'
    file_path = os.path.join(reports_dir, filename)
    
    if os.path.exists(file_path) and filename.endswith('.html'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/html'}
    else:
        return "Report not found", 404

def adapt_and_validate_corpus_file(file_path):
    """
    Adapts a corpus file to the required internal format and validates it.
    This function reads the file, adapts each line in memory, and then
    overwrites the original file with the corrected data.
    """
    adapted_records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Skip empty lines
                if not line.strip():
                    continue

                try:
                    original_record = json.loads(line)
                    
                    # --- Adaptation Logic ---
                    adapted_record = {
                        # Use 'corpus_item_id' as 'id'
                        'id': original_record.get('corpus_item_id'),
                        # Add a timestamp, as it's missing
                        'timestamp': datetime.utcnow().isoformat(),
                        # Get 'platform' from the nested 'source_details'
                        'platform': original_record.get('source_details', {}).get('platform'),
                        # --- Carry over other original fields ---
                        'version': original_record.get('version'),
                        'source_details': original_record.get('source_details'),
                        'original_content': original_record.get('original_content'),
                        'llm_transformation': original_record.get('llm_transformation')
                    }
                    
                    # --- Validation Logic (on the adapted record) ---
                    if not all([adapted_record['id'], adapted_record['timestamp'], adapted_record['platform']]):
                        return False, f"Missing required data in line {i+1}. Could not find id, timestamp, or platform."

                    adapted_records.append(adapted_record)

                except json.JSONDecodeError:
                    return False, f"Invalid JSON on line {i+1}"
                except KeyError as e:
                    return False, f"Missing key {e} in line {i+1}"
        
        if not adapted_records:
            return False, "File is empty or contains no valid JSON."

        # If all lines are processed successfully, overwrite the file with adapted data
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in adapted_records:
                f.write(json.dumps(record) + '\n')
                
        return True, "File adapted and validated successfully"

    except Exception as e:
        return False, f"Error reading or processing file: {str(e)}"

@app.route('/upload', methods=['POST'])
def upload_corpus():
    """Handles corpus file uploads."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    file = request.files['file']
    corpus_type = request.form.get('corpus_type', 'original')
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if not file.filename.endswith('.jsonl'):
        return jsonify({'success': False, 'message': 'Only .jsonl files are allowed'}), 400
    
    # Secure the filename
    filename = secure_filename(file.filename)
    
    # Determine upload directory
    if corpus_type == 'rewritten':
        upload_dir = 'corpora/rewritten_pairs'
    else:
        upload_dir = 'corpora/original_only'
    
    # Create directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save file temporarily for validation
    temp_path = os.path.join(upload_dir, f"temp_{filename}")
    file.save(temp_path)
    
    # Adapt and validate the file
    is_valid, message = adapt_and_validate_corpus_file(temp_path)
    
    if not is_valid:
        # Remove invalid file
        os.remove(temp_path)
        return jsonify({'success': False, 'message': f'Invalid corpus file: {message}'}), 400
    
    # Move to final location
    final_path = os.path.join(upload_dir, filename)
    
    # Check if file already exists
    if os.path.exists(final_path):
        os.remove(temp_path)
        return jsonify({'success': False, 'message': 'File already exists'}), 400
    
    os.rename(temp_path, final_path)
    
    return jsonify({
        'success': True, 
        'message': f'Corpus file uploaded successfully to {corpus_type} collection',
        'filename': filename,
        'path': final_path
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
