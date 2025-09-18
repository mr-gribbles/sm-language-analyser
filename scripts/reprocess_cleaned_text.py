#!/usr/bin/env python3
"""
Script to reprocess JSONL files and update cleaned_text fields with improved URL cleaning.

This script handles different JSONL file formats used in the social media predictor project:
- Test files (test_data_human.jsonl, test_data_ai.jsonl)
- Corpus files (original_only, rewritten_pairs, llm_generated_social_media)

It updates all relevant text fields using the improved data_cleaner.py that removes URLs properly.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.data_cleaner import clean_text
    print("✓ Successfully imported improved clean_text function")
except ImportError as e:
    print(f"✗ Error importing clean_text: {e}")
    sys.exit(1)


def backup_file(file_path: Path) -> Path:
    """Create a backup of the original file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.with_suffix(f'.backup_{timestamp}{file_path.suffix}')
    shutil.copy2(file_path, backup_path)
    print(f"  Backup created: {backup_path.name}")
    return backup_path


def clean_text_field(text: Optional[str]) -> Optional[str]:
    """Clean a text field if it exists and is valid."""
    if text and isinstance(text, str) and text.strip():
        try:
            cleaned = clean_text(text)
            return cleaned if cleaned.strip() else text
        except Exception as e:
            print(f"  Warning: Error cleaning text: {e}")
            return text
    return text


def process_original_content(original_content: Dict[str, Any]) -> Dict[str, Any]:
    """Process and clean text fields in original_content section."""
    if not isinstance(original_content, dict):
        return original_content
    
    # Clean various text fields that might exist across different formats
    text_fields = [
        'cleaned_text', 'cleaned_selftext',     # Bluesky/Reddit cleaned text
        'raw_text', 'raw_selftext',             # Bluesky/Reddit raw text
        'title',                                # Post titles
        'content',                              # LLM generated content field
        'author',                               # Author field (might contain URLs)
        'url'                                   # Direct URL fields
    ]
    cleaned_fields = []
    
    for field in text_fields:
        if field in original_content:
            original_text = original_content[field]
            cleaned_text = clean_text_field(original_text)
            if cleaned_text != original_text:
                original_content[field] = cleaned_text
                cleaned_fields.append(field)
    
    if cleaned_fields:
        print(f"    Cleaned original_content fields: {', '.join(cleaned_fields)}")
    
    return original_content


def process_llm_transformation(llm_transformation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Process and clean text fields in llm_transformation section."""
    if not llm_transformation or not isinstance(llm_transformation, dict):
        return llm_transformation
    
    # Clean rewritten text
    if 'rewritten_text' in llm_transformation:
        original_text = llm_transformation['rewritten_text']
        cleaned_text = clean_text_field(original_text)
        if cleaned_text != original_text:
            llm_transformation['rewritten_text'] = cleaned_text
            print(f"    Cleaned llm_transformation.rewritten_text")
    
    return llm_transformation


def process_top_level_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process and clean any text fields that might exist at the top level."""
    if not isinstance(data, dict):
        return data
    
    # Fields that might contain text/URLs at the top level (especially for LLM generated)
    top_level_text_fields = [
        'source',           # Source field might have URLs
        'id',               # ID might contain URL fragments  
        'post_id',          # Post ID might contain URLs
        'url'               # Direct URL fields
    ]
    
    cleaned_fields = []
    
    for field in top_level_text_fields:
        if field in data:
            original_text = data[field]
            cleaned_text = clean_text_field(original_text)
            if cleaned_text != original_text:
                data[field] = cleaned_text
                cleaned_fields.append(field)
    
    # Also process nested dictionaries that might contain text (metadata, processing_info, etc.)
    nested_dicts = ['metadata', 'processing_info', 'source_details']
    
    for nested_field in nested_dicts:
        if nested_field in data and isinstance(data[nested_field], dict):
            original_nested = json.dumps(data[nested_field], sort_keys=True)
            data[nested_field] = process_nested_dict(data[nested_field], nested_field)
            modified_nested = json.dumps(data[nested_field], sort_keys=True)
            if original_nested != modified_nested:
                cleaned_fields.append(nested_field)
    
    if cleaned_fields:
        print(f"    Cleaned top-level fields: {', '.join(cleaned_fields)}")
    
    return data


def process_nested_dict(nested_data: Dict[str, Any], section_name: str) -> Dict[str, Any]:
    """Process nested dictionaries for text fields that might contain URLs."""
    if not isinstance(nested_data, dict):
        return nested_data
    
    # Common fields in metadata/processing_info that might contain URLs
    text_fields = [
        'post_url',         # Direct post URLs
        'url',              # Any URL field
        'platform',         # Platform names might have URL fragments
        'subreddit',        # Subreddit names might have URL parts
        'author_handle',    # Author handles might contain domains
        'source_type'       # Source types might contain URL fragments
    ]
    
    cleaned_fields = []
    
    for field in text_fields:
        if field in nested_data:
            original_text = nested_data[field]
            cleaned_text = clean_text_field(original_text)
            if cleaned_text != original_text:
                nested_data[field] = cleaned_text
                cleaned_fields.append(field)
    
    if cleaned_fields:
        print(f"    Cleaned {section_name} fields: {', '.join(cleaned_fields)}")
    
    return nested_data


def process_jsonl_file(file_path: Path, dry_run: bool = False) -> Dict[str, int]:
    """
    Process a JSONL file and update cleaned text fields.
    
    Args:
        file_path: Path to the JSONL file
        dry_run: If True, show what would be changed without modifying files
        
    Returns:
        Dictionary with processing statistics
    """
    print(f"\nProcessing: {file_path}")
    
    if not file_path.exists():
        print(f"  ✗ File not found: {file_path}")
        return {'processed': 0, 'modified': 0, 'errors': 0}
    
    stats = {'processed': 0, 'modified': 0, 'errors': 0}
    
    try:
        # Read all lines
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"  ✗ File is empty")
            return stats
        
        modified_lines = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                modified_lines.append('')
                continue
            
            try:
                # Parse JSON
                data = json.loads(line)
                original_data = json.dumps(data, sort_keys=True)
                
                # Process top-level fields and nested dictionaries
                data = process_top_level_fields(data)
                
                # Process original_content
                if 'original_content' in data:
                    data['original_content'] = process_original_content(data['original_content'])
                
                # Process llm_transformation
                if 'llm_transformation' in data:
                    data['llm_transformation'] = process_llm_transformation(data['llm_transformation'])
                
                # Check if anything changed
                modified_data = json.dumps(data, sort_keys=True)
                if original_data != modified_data:
                    stats['modified'] += 1
                
                modified_lines.append(json.dumps(data, ensure_ascii=False))
                stats['processed'] += 1
                
            except json.JSONDecodeError as e:
                print(f"  ✗ JSON error on line {line_num}: {e}")
                modified_lines.append(line)
                stats['errors'] += 1
            except Exception as e:
                print(f"  ✗ Processing error on line {line_num}: {e}")
                modified_lines.append(line)
                stats['errors'] += 1
        
        # Write results if not dry run
        if not dry_run and stats['modified'] > 0:
            # Create backup
            backup_path = backup_file(file_path)
            
            # Write modified file
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in modified_lines:
                    f.write(line + '\n')
            
            print(f"  ✓ Updated {stats['modified']} entries (backup: {backup_path.name})")
        elif dry_run and stats['modified'] > 0:
            print(f"  [DRY RUN] Would update {stats['modified']} entries")
        else:
            print(f"  ✓ No changes needed")
        
        print(f"  Stats: {stats['processed']} processed, {stats['modified']} modified, {stats['errors']} errors")
        
    except Exception as e:
        print(f"  ✗ Error processing file: {e}")
        stats['errors'] += 1
    
    return stats


def find_jsonl_files(base_dir: Path) -> list[Path]:
    """Find all JSONL files to process."""
    jsonl_files = []
    
    # Test files
    test_files = ['test_data_human.jsonl', 'test_data_ai.jsonl']
    for filename in test_files:
        file_path = base_dir / filename
        if file_path.exists():
            jsonl_files.append(file_path)
    
    # Corpus files
    corpora_dir = base_dir / 'corpora'
    if corpora_dir.exists():
        for jsonl_file in corpora_dir.rglob('*.jsonl'):
            jsonl_files.append(jsonl_file)
    
    return sorted(jsonl_files)


def main():
    """Main function to reprocess all JSONL files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reprocess JSONL files with improved URL cleaning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be changed
  python scripts/reprocess_cleaned_text.py --dry-run
  
  # Process all files
  python scripts/reprocess_cleaned_text.py
  
  # Process specific files
  python scripts/reprocess_cleaned_text.py test_data_human.jsonl corpora/original_only/*.jsonl
        """
    )
    
    parser.add_argument('files', nargs='*', 
                       help='Specific JSONL files to process (default: all files)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    
    args = parser.parse_args()
    
    # Get base directory
    base_dir = Path(__file__).parent.parent
    
    print("=" * 80)
    print("JSONL FILE REPROCESSING WITH IMPROVED URL CLEANING")
    print("=" * 80)
    
    # Determine which files to process
    if args.files:
        jsonl_files = []
        for file_pattern in args.files:
            file_path = Path(file_pattern)
            if not file_path.is_absolute():
                file_path = base_dir / file_path
            
            if file_path.exists():
                jsonl_files.append(file_path)
            else:
                print(f"Warning: File not found: {file_path}")
    else:
        jsonl_files = find_jsonl_files(base_dir)
    
    if not jsonl_files:
        print("No JSONL files found to process.")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process:")
    for file_path in jsonl_files:
        rel_path = file_path.relative_to(base_dir)
        print(f"  - {rel_path}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] - No files will be modified")
    
    # Process files
    total_stats = {'processed': 0, 'modified': 0, 'errors': 0}
    
    for file_path in jsonl_files:
        stats = process_jsonl_file(file_path, dry_run=args.dry_run)
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total entries processed: {total_stats['processed']}")
    print(f"Total entries modified: {total_stats['modified']}")
    print(f"Total errors: {total_stats['errors']}")
    
    if not args.dry_run and total_stats['modified'] > 0:
        print(f"\n✓ Successfully updated {total_stats['modified']} entries across {len(jsonl_files)} files")
        print("  Backup files created for all modified files")
        print("\nNext steps:")
        print("  1. Retrain your models with the cleaned data")
        print("  2. Re-run interpretability analysis to verify 'www' is eliminated")
    elif args.dry_run:
        print(f"\n[DRY RUN] Would modify {total_stats['modified']} entries")
        print("Run without --dry-run to apply changes")


if __name__ == '__main__':
    main()
