"""
Data preprocessing script for T5 text-to-SQL task.
Applies: 1) Text normalization, 2) SQL formatting, 3) Schema appending, 4) Data cleaning
"""

import re
import os
import json
from typing import List, Tuple
from transformers import T5TokenizerFast
from collections import Counter

from sql_processing import preprocess_sql

# Load tokenizer
tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

# ============= Schema Functions =============

def load_schema(schema_path: str) -> str:
    """
    Load database schema and extract only table names (not column names).
    This keeps the token length manageable for T5.
    Format: table1 | table2 | table3 (using | as separator)
    """
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    # Extract table names from the "ents" section
    table_names = list(schema.get('ents', {}).keys())

    # Create a simplified schema string with just table names, separated by |
    schema_text = " | ".join(table_names)

    return schema_text

def append_schema_to_nl(nl_text: str, schema_text: str) -> str:
    """
    Append simplified schema information to natural language input.
    Format: translate to SQL: [nl_text]\ntables: [schema_text]
    """
    return f"translate to SQL: {nl_text}\ntables: {schema_text}"

# ============= Preprocessing Functions =============

def normalize_text(text: str) -> str:
    """
    1. Text Normalization:
       - Convert to lowercase
       - Remove extra whitespace/newlines
       - Normalize punctuation spacing
    """
    # Convert to lowercase
    text = text.lower()

    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove extra spaces around punctuation
    text = re.sub(r'\s([.,!?;:])', r'\1', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def normalize_sql(query: str) -> str:
    """
    2. SQL Formatting (delegates to sql_processing.preprocess_sql):
       - Space-pad operators/punctuation for consistent tokenization
       - Lowercase
       - Compress ATIS table aliases (flight_1 -> f1, airport_service_1 -> as1, ...)
       - Normalize quotes
    """
    query = query.replace('"', "'")
    return preprocess_sql(query)


def is_valid_example(nl: str, sql: str, max_nl_length: int = 256, max_sql_length: int = 512) -> bool:
    """
    4. Data Cleaning:
       - Remove examples with extremely long sequences
       - Remove examples with very short or empty text
    """
    # Check if text is empty
    if not nl.strip() or not sql.strip():
        return False

    # Tokenize to check lengths
    nl_tokens = tokenizer.encode(nl, max_length=1000, truncation=False)
    sql_tokens = tokenizer.encode(sql, max_length=1000, truncation=False)

    # Filter by token length
    if len(nl_tokens) > max_nl_length or len(sql_tokens) > max_sql_length:
        return False

    return True


def preprocess_data(nl_list: List[str], sql_list: List[str], schema_path: str = 'data/flight_database.schema') -> Tuple[List[str], List[str]]:
    """
    Apply all preprocessing steps to the data.
    Includes: normalization, schema appending, and validation.
    Handles test split where sql_list may contain None values.
    """
    processed_nl = []
    processed_sql = []

    # Load simplified schema (table names only)
    schema_text = load_schema(schema_path)

    num_removed = 0
    for nl, sql in zip(nl_list, sql_list):
        # Normalize
        nl_norm = normalize_text(nl)

        # Handle test split where sql is None
        if sql is None:
            sql_norm = None
            # For test set, just validate NL length
            if nl_norm.strip():  # Not empty
                nl_with_schema = nl_norm
                # append_schema_to_nl(nl_norm, schema_text)
                # nl_with_schema = append_schema_to_nl(nl_norm, schema_text)
                processed_nl.append(nl_with_schema)
                processed_sql.append(sql_norm)
            else:
                num_removed += 1
        else:
            sql_norm = normalize_sql(sql)

            # Append schema to NL (reduces token length by only including table names)
            nl_with_schema = nl_norm
            # append_schema_to_nl(nl_norm, schema_text)

            # Clean
            if is_valid_example(nl_with_schema, sql_norm):
                processed_nl.append(nl_with_schema)
                processed_sql.append(sql_norm)
            else:
                num_removed += 1

    print(f"Removed {num_removed} examples during data cleaning")
    return processed_nl, processed_sql


# ============= Statistics Functions =============

def compute_statistics(nl_list: List[str], sql_list: List[str]) -> dict:
    """
    Compute data statistics using T5 tokenizer.
    """
    stats = {}

    # Number of examples
    stats['num_examples'] = len(nl_list)

    # Tokenize all examples
    nl_lengths = []
    sql_lengths = []
    all_nl_tokens = []
    all_sql_tokens = []

    for nl, sql in zip(nl_list, sql_list):
        nl_tokens = tokenizer.encode(nl)
        sql_tokens = tokenizer.encode(sql)

        nl_lengths.append(len(nl_tokens))
        sql_lengths.append(len(sql_tokens))

        all_nl_tokens.extend(nl_tokens)
        all_sql_tokens.extend(sql_tokens)

    # Mean lengths
    stats['mean_nl_length'] = sum(nl_lengths) / len(nl_lengths) if nl_lengths else 0
    stats['max_nl_length'] = max(nl_lengths) if nl_lengths else 0
    stats['min_nl_length'] = min(nl_lengths) if nl_lengths else 0

    stats['mean_sql_length'] = sum(sql_lengths) / len(sql_lengths) if sql_lengths else 0
    stats['max_sql_length'] = max(sql_lengths) if sql_lengths else 0
    stats['min_sql_length'] = min(sql_lengths) if sql_lengths else 0

    # Vocabulary sizes
    stats['vocab_nl'] = len(set(all_nl_tokens))
    stats['vocab_sql'] = len(set(all_sql_tokens))

    return stats


def load_data(split: str) -> Tuple[List[str], List[str]]:
    """Load raw data from files."""
    nl_path = f'data/{split}.nl'
    sql_path = f'data/{split}.sql'

    with open(nl_path, 'r') as f:
        nl_list = [line.strip() for line in f.readlines()]

    # SQL file may not exist for test split
    if os.path.exists(sql_path):
        with open(sql_path, 'r') as f:
            sql_list = [line.strip() for line in f.readlines()]
    else:
        sql_list = [None] * len(nl_list)  # No SQL labels for test set

    return nl_list, sql_list


def print_statistics_table(stats_before: dict, stats_after: dict, split: str):
    """Print statistics in table format."""
    print(f"\n{'='*80}")
    print(f"Statistics for {split.upper()} set")
    print(f"{'='*80}\n")

    print(f"{'Statistic':<40} {'Before Preprocessing':<25} {'After Preprocessing':<25}")
    print("-" * 90)

    print(f"{'Number of examples':<40} {stats_before['num_examples']:<25} {stats_after['num_examples']:<25}")
    print(f"{'Mean NL length (tokens)':<40} {stats_before['mean_nl_length']:<25.2f} {stats_after['mean_nl_length']:<25.2f}")
    print(f"{'Max NL length (tokens)':<40} {stats_before['max_nl_length']:<25} {stats_after['max_nl_length']:<25}")
    print(f"{'Min NL length (tokens)':<40} {stats_before['min_nl_length']:<25} {stats_after['min_nl_length']:<25}")
    print(f"{'Mean SQL length (tokens)':<40} {stats_before['mean_sql_length']:<25.2f} {stats_after['mean_sql_length']:<25.2f}")
    print(f"{'Max SQL length (tokens)':<40} {stats_before['max_sql_length']:<25} {stats_after['max_sql_length']:<25}")
    print(f"{'Min SQL length (tokens)':<40} {stats_before['min_sql_length']:<25} {stats_after['min_sql_length']:<25}")
    print(f"{'NL Vocabulary size':<40} {stats_before['vocab_nl']:<25} {stats_after['vocab_nl']:<25}")
    print(f"{'SQL Vocabulary size':<40} {stats_before['vocab_sql']:<25} {stats_after['vocab_sql']:<25}")
    print()


def main():
    """Main preprocessing pipeline."""
    print("Loading data...")

    # Load training, dev, and test data
    train_nl, train_sql = load_data('train')
    dev_nl, dev_sql = load_data('dev')
    test_nl, test_sql = load_data('test')

    print(f"Loaded {len(train_nl)} training examples")
    print(f"Loaded {len(dev_nl)} dev examples")
    print(f"Loaded {len(test_nl)} test examples")

    # Compute statistics BEFORE preprocessing
    print("\nComputing statistics before preprocessing...")
    train_stats_before = compute_statistics(train_nl, train_sql)
    dev_stats_before = compute_statistics(dev_nl, dev_sql)

    # Apply preprocessing
    print("\nApplying preprocessing steps...")
    train_nl_proc, train_sql_proc = preprocess_data(train_nl, train_sql)
    dev_nl_proc, dev_sql_proc = preprocess_data(dev_nl, dev_sql)
    test_nl_proc, test_sql_proc = preprocess_data(test_nl, test_sql)

    # Compute statistics AFTER preprocessing
    print("Computing statistics after preprocessing...")
    train_stats_after = compute_statistics(train_nl_proc, train_sql_proc)
    dev_stats_after = compute_statistics(dev_nl_proc, dev_sql_proc)

    # Print results
    print_statistics_table(train_stats_before, train_stats_after, 'train')
    print_statistics_table(dev_stats_before, dev_stats_after, 'dev')

    # Save preprocessed data
    print("Saving preprocessed data...")
    os.makedirs('data_preprocessed', exist_ok=True)

    with open('data_preprocessed/train.nl', 'w') as f:
        f.write('\n'.join(train_nl_proc))

    with open('data_preprocessed/train.sql', 'w') as f:
        f.write('\n'.join(train_sql_proc))

    with open('data_preprocessed/dev.nl', 'w') as f:
        f.write('\n'.join(dev_nl_proc))

    with open('data_preprocessed/dev.sql', 'w') as f:
        f.write('\n'.join(dev_sql_proc))

    with open('data_preprocessed/test.nl', 'w') as f:
        f.write('\n'.join(test_nl_proc))

    print("✓ Preprocessed data saved to data_preprocessed/")

    # Print summary for table 2
    print("\n" + "="*80)
    print("TABLE 1 - BEFORE PREPROCESSING")
    print("="*80)
    print(f"Train: {train_stats_before['num_examples']} | Mean NL: {train_stats_before['mean_nl_length']:.2f} | Mean SQL: {train_stats_before['mean_sql_length']:.2f} | NL Vocab: {train_stats_before['vocab_nl']} | SQL Vocab: {train_stats_before['vocab_sql']}")
    print(f"Dev:   {dev_stats_before['num_examples']} | Mean NL: {dev_stats_before['mean_nl_length']:.2f} | Mean SQL: {dev_stats_before['mean_sql_length']:.2f} | NL Vocab: {dev_stats_before['vocab_nl']} | SQL Vocab: {dev_stats_before['vocab_sql']}")

    print("\n" + "="*80)
    print("TABLE 2 - AFTER PREPROCESSING")
    print("="*80)
    print(f"Train: {train_stats_after['num_examples']} | Mean NL: {train_stats_after['mean_nl_length']:.2f} | Mean SQL: {train_stats_after['mean_sql_length']:.2f} | NL Vocab: {train_stats_after['vocab_nl']} | SQL Vocab: {train_stats_after['vocab_sql']}")
    print(f"Dev:   {dev_stats_after['num_examples']} | Mean NL: {dev_stats_after['mean_nl_length']:.2f} | Mean SQL: {dev_stats_after['mean_sql_length']:.2f} | NL Vocab: {dev_stats_after['vocab_nl']} | SQL Vocab: {dev_stats_after['vocab_sql']}")


if __name__ == '__main__':
    main()

