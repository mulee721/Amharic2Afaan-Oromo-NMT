import os
import pandas as pd
import numpy as np
import sentencepiece as spm
from collections import Counter

# File paths specified from your environment
MODEL_PATH = r"C:\Users\mulee\Documents\AI-MT\AI-MT\tokenizer\translator_sp.model"
VOCAB_PATH = r"C:\Users\mulee\Documents\AI-MT\AI-MT\tokenizer\translator_sp.vocab"
TEST_CSV_PATH = r"C:\Users\mulee\Documents\AI-MT\AI-MT\data\processed\test.csv"
REPORT_OUTPUT_PATH = r"C:\Users\mulee\Documents\AI-MT\AI-MT\tokenizer\evaluation_report.txt"

def evaluate_tokenizer():
    report_lines = []

    def log_and_print(text=""):
        """Prints to console and appends to the report lines list simultaneously."""
        print(text)
        report_lines.append(str(text))

    log_and_print("=" * 60)
    log_and_print("SENTENCEPIECE TOKENIZER EVALUATION REPORT")
    log_and_print("=" * 60)
    
    # 1. Load SentencePiece Model
    if not os.path.exists(MODEL_PATH):
        log_and_print(f"Error: Model file not found at {MODEL_PATH}")
        return
        
    sp = spm.SentencePieceProcessor()
    sp.load(MODEL_PATH)
    total_vocab_size = sp.get_piece_size()
    unk_id = sp.unk_id()
    log_and_print(f"Successfully loaded model. Total Vocabulary Size: {total_vocab_size}")
    
    # 2. Load Test Data
    if not os.path.exists(TEST_CSV_PATH):
        log_and_print(f"Error: Test CSV not found at {TEST_CSV_PATH}")
        return
        
    df = pd.read_csv(TEST_CSV_PATH)
    log_and_print(f"Loaded test dataset. Shape: {df.shape}")
    log_and_print(f"Columns found in CSV: {list(df.columns)}")
    
    # Dynamically detect columns for Amharic and Afaan Oromo
    col_am = next((c for c in df.columns if 'am' in c.lower() or 'amharic' in c.lower()), df.columns[0])
    col_om = next((c for c in df.columns if 'or' in c.lower() or 'om' in c.lower() or 'afaan' in c.lower()), df.columns[1])
    
    log_and_print(f"Mapping -> Amharic Column: '{col_am}' | Afaan Oromo Column: '{col_om}'\n")
    
    am_sentences = df[col_am].dropna().astype(str).tolist()
    om_sentences = df[col_om].dropna().astype(str).tolist()
    
    # 3. Analysis Function per Language
    def analyze_language(sentences, lang_name):
        token_counts = []
        word_counts = []
        unk_count = 0
        total_tokens = 0
        used_ids = set()
        all_pieces = []
        
        for sent in sentences:
            words = sent.split()
            word_counts.append(len(words))
            
            pieces = sp.encode_as_pieces(sent)
            ids = sp.encode_as_ids(sent)
            
            token_counts.append(len(pieces))
            total_tokens += len(pieces)
            
            for pid in ids:
                used_ids.add(pid)
                if pid == unk_id:
                    unk_count += 1
                    
            all_pieces.extend(pieces)
            
        token_counts = np.array(token_counts)
        word_counts = np.array(word_counts)
        
        log_and_print(f"--- {lang_name} Tokenization Metrics ---")
        log_and_print(f"* Total Sentences Evaluated: {len(sentences):,}")
        log_and_print(f"* Tokens/Sentence:")
        log_and_print(f"    - Mean   : {token_counts.mean():.2f}")
        log_and_print(f"    - Median : {np.median(token_counts):.1f}")
        log_and_print(f"    - P95    : {np.percentile(token_counts, 95):.1f}")
        log_and_print(f"    - Max    : {token_counts.max()}")
        log_and_print(f"* Words/Sentence (Mean)  : {word_counts.mean():.2f}")
        
        avg_words = word_counts.mean() if word_counts.mean() > 0 else 1
        avg_tokens = token_counts.mean()
        compression_ratio = avg_tokens / avg_words
        log_and_print(f"* Compression Ratio (Subwords/Words) : {compression_ratio:.2f}")
        
        unk_rate = (unk_count / total_tokens) * 100 if total_tokens > 0 else 0
        log_and_print(f"* Unknown Token (<unk>) Rate         : {unk_rate:.4f}% ({unk_count:,} <unk> in {total_tokens:,} tokens)")
        
        corpus_vocab_util = len(used_ids)
        log_and_print(f"* Corpus Vocabulary Utilization      : {corpus_vocab_util} / {total_vocab_size} pieces used ({(corpus_vocab_util/total_vocab_size)*100:.2f}%)")
        
        log_and_print(f"* Top 10 Most Frequent Subwords:")
        counter = Counter(all_pieces)
        for piece, freq in counter.most_common(10):
            log_and_print(f"    - '{piece}': {freq:,} times")
        log_and_print("-" * 60 + "\n")

    # Run analysis for both languages
    analyze_language(am_sentences, "Amharic")
    analyze_language(om_sentences, "Afaan Oromo")

    # 4. Save Report to Text File
    try:
        os.makedirs(os.path.dirname(REPORT_OUTPUT_PATH), exist_ok=True)
        with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n[Success] Evaluation report successfully saved to: {REPORT_OUTPUT_PATH}")
    except Exception as e:
        print(f"\n[Error] Failed to save report file: {e}")

if __name__ == "__main__":
    evaluate_tokenizer()