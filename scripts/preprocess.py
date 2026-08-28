from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REMOVED_DIR = DATA_DIR / "removed"

INPUT_CSV_FILE = DATA_DIR / "classified_am_to_ao.csv"

FINAL_CSV_FILE = DATA_DIR / "final.csv"

TRAIN_CSV_FILE = PROCESSED_DIR / "train.csv"
VALID_CSV_FILE = PROCESSED_DIR / "valid.csv"
TEST_CSV_FILE = PROCESSED_DIR / "test.csv"

REPORT_FILE = PROCESSED_DIR / "preprocess_report.json"


SOURCE_COLUMN = "Amharic"
TARGET_COLUMN = "Oromo"
DOMAIN_COLUMN = "Domain"

DATA_COLUMNS = [SOURCE_COLUMN, TARGET_COLUMN]


RANDOM_STATE = 42

TEST_SIZE = 0.20
VALID_SIZE = 0.50


MIN_WORDS = 1
MAX_WORDS = 80

MAX_CHARS = 250

# Maximum ratio threshold for Amharic (fusional) to Afaan Oromo (agglutinative) expansion
MAX_RATIO = 5.0


USE_AMHARIC_ORTHOGRAPHY_MAP = True
NORMALIZE_OROMO_PUNCTUATION = True

REMOVE_DUPLICATES = True
REMOVE_NEAR_DUPLICATES = False
PRESERVE_CASE = True


AMHARIC_CHARACTER_MAP = str.maketrans(
    {
        "ሐ": "ሀ", "ኀ": "ሀ", "ሓ": "ሃ", "ኃ": "ሃ", "ሑ": "ሁ", "ኁ": "ሁ", "ሒ": "ሂ",
        "ኂ": "ሂ", "ሔ": "ሄ", "ኄ": "ሄ", "ሕ": "ህ", "ኅ": "ህ", "ሖ": "ሆ", "ኆ": "ሆ", 
        "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ", "ዐ": "አ",
        "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ", "ፀ": "ጸ", "ፁ": "ጹ",
        "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
    }
)

# --- PUNCTUATION MAPPING PER LANGUAGE ---
# For Amharic (Source): Standardize punctuation to Ethiopic formats (converting standard ASCII to Ethiopic)
AMHARIC_PUNCTUATION_MAP = str.maketrans(
    {
        ".": "።",  # Period to Ethiopic Full Stop
        ",": "፣",  # Comma to Ethiopic Comma
        ":": "፥",  # Colon to Ethiopic Colon
        ";": "፤",  # Semicolon to Ethiopic Semicolon
        "?": "፧",  # Question Mark to Ethiopic Question Mark
    }
)

# For Afaan Oromo (Target): Map stray Ethiopic symbols and smart quotes to standard ASCII
OROMO_PUNCTUATION_MAP = str.maketrans(
    {
        "።": ".",  # Stray Ethiopic Full Stop to Period
        "፣": ",",  # Stray Ethiopic Comma to Comma
        "፥": ":",  # Stray Ethiopic Colon to Colon
        "፤": ";",  # Stray Ethiopic Semicolon to Semicolon
        "፧": "?",  # Stray Ethiopic Question Mark to Question Mark
        "፡": " ",  # Stray Ethiopic Separator to Space
        "፦": "",   # Stray Ethiopic Paragraph Separator (Remove)
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


PUNCTUATION_SPACING_RE = re.compile(
    r"\s+([?.!,;:።፣፤፥፦፧])"
)

MULTI_PUNCTUATION_RE = re.compile(
    r"([?.!,;:።፣፤፥፦፧])\1{2,}"
)

WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(
    r"[A-Za-z0-9\u1200-\u137F]+"
)


HTML_RE = re.compile(
    r"<[^>]+>"
)

URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+",
    re.IGNORECASE,
)

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

ARABIC_SCRIPT_RE = re.compile(
    r"[\u0600-\u06FF"
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"\uFB50-\uFDFF"
    r"\uFE70-\uFEFF]"
)

CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

REPLACEMENT_CHAR_RE = re.compile(
    "\uFFFD"
)

EXCESSIVE_PUNCTUATION_RE = re.compile(
    r"([!?.,;:])\1{3,}"
)


@dataclass
class PreprocessStats:
    raw_rows: int = 0
    after_column_selection: int = 0
    after_normalization: int = 0
    after_dropna: int = 0
    removed_empty: int = 0
    after_deduplication: int = 0
    removed_duplicates: int = 0
    after_quality_filter: int = 0
    removed_quality: int = 0
    removed_length: int = 0
    removed_ratio: int = 0
    removed_characters: int = 0
    removed_script: int = 0
    removed_html: int = 0
    removed_url: int = 0
    removed_email: int = 0
    removed_arabic_script: int = 0
    removed_control_chars: int = 0
    removed_replacement_chars: int = 0
    removed_excessive_punctuation: int = 0
    final_rows: int = 0


def ensure_directories() -> None:
    """Create required project directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REMOVED_DIR.mkdir(parents=True, exist_ok=True)


def remove_existing_outputs() -> None:
    """Remove generated preprocessing outputs from previous runs."""
    output_files = [
        FINAL_CSV_FILE,
        TRAIN_CSV_FILE,
        VALID_CSV_FILE,
        TEST_CSV_FILE,
        REPORT_FILE,
    ]

    if REMOVED_DIR.exists():
        output_files.extend(list(REMOVED_DIR.glob("*.csv")))

    removed = []
    for path in output_files:
        if path.exists():
            path.unlink()
            removed.append(path)

    if removed:
        print("\n🧹 Removed previous generated outputs:")
        for path in removed:
            print(f"   ✓ {path}")


def load_csv_input(path: Path) -> pd.DataFrame:
    """Load the input CSV and validate that it exists."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV file not found:\n{path}")

    print(f"\n📂 Loading input file:")
    print(f"   {path}")

    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [str(column).strip() for column in df.columns]
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as UTF-8 CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def save_removed_csv(dataset: pd.DataFrame | list, file_name: str) -> None:
    """Helper to save filtered rows to data/removed/ if any exist."""
    if isinstance(dataset, list):
        if len(dataset) > 0:
            df = pd.DataFrame(dataset)
            save_csv(df, REMOVED_DIR / file_name)
    elif not dataset.empty:
        save_csv(dataset, REMOVED_DIR / file_name)


def save_report(report: dict, path: Path) -> None:
    """Save preprocessing information as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def require_columns(df: pd.DataFrame) -> None:
    """Ensure required translation columns exist."""
    missing = set(DATA_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def count_words(text: object) -> int:
    """Count actual words recognizing Latin and Ethiopic characters."""
    if pd.isna(text):
        return 0
    text = str(text)
    return len(WORD_RE.findall(text))


def normalize_text(
    value: object,
    orthography_map: dict | None = None,
    punctuation_map: dict | None = None,
) -> str:
    """Normalize text with sequence handling, maps, and regex formatting."""
    if pd.isna(value):
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKC", text)

    # First handle multi-character sequence replacements
    text = text.replace("።።", "።")
    text = text.replace("\u200b", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\t", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    if orthography_map is not None:
        text = text.translate(orthography_map)

    if punctuation_map is not None:
        text = text.translate(punctuation_map)

    text = text.replace('""', '"')
    text = text.replace("''", "'")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("…", "...")

    text = PUNCTUATION_SPACING_RE.sub(r"\1", text)
    text = MULTI_PUNCTUATION_RE.sub(r"\1", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    if not PRESERVE_CASE:
        text = text.lower()

    return text


def normalize_amharic(value: object) -> str:
    """Normalize Amharic source text using Ethiopic mappings."""
    orthography_map = AMHARIC_CHARACTER_MAP if USE_AMHARIC_ORTHOGRAPHY_MAP else None
    return normalize_text(
        value,
        orthography_map=orthography_map,
        punctuation_map=AMHARIC_PUNCTUATION_MAP,
    )


def normalize_oromo(value: object) -> str:
    """Normalize Afaan Oromo target text using Latin & noise cleanup mappings."""
    punctuation_map = OROMO_PUNCTUATION_MAP if NORMALIZE_OROMO_PUNCTUATION else None
    return normalize_text(
        value,
        orthography_map=None,
        punctuation_map=punctuation_map,
    )


def has_valid_script(text: str, source: bool) -> bool:
    """Check whether a sentence contains expected script characters."""
    if not text:
        return False

    if source:
        return any("\u1200" <= character <= "\u137F" for character in text)

    return any(character.isalpha() for character in text)


def detect_noise(source: str, target: str) -> list[str]:
    """Detect common corpus noise."""
    combined = f"{source} {target}"
    reasons: list[str] = []

    if HTML_RE.search(combined):
        reasons.append("html")
    if URL_RE.search(combined):
        reasons.append("url")
    if EMAIL_RE.search(combined):
        reasons.append("email")
    if ARABIC_SCRIPT_RE.search(combined):
        reasons.append("arabic_script")
    if CONTROL_CHAR_RE.search(combined):
        reasons.append("control_chars")
    if REPLACEMENT_CHAR_RE.search(combined):
        reasons.append("replacement_chars")
    if EXCESSIVE_PUNCTUATION_RE.search(combined):
        reasons.append("excessive_punctuation")

    return reasons


def evaluate_row(source: str, target: str) -> tuple[bool, str]:
    """Validate one translation pair."""
    if not source or not target:
        return False, "empty"

    noise = detect_noise(source, target)
    if noise:
        return False, "noise:" + ",".join(noise)

    if len(source) > MAX_CHARS or len(target) > MAX_CHARS:
        return False, "characters"

    source_words = count_words(source)
    target_words = count_words(target)

    if source_words < MIN_WORDS or target_words < MIN_WORDS:
        return False, "length"

    if source_words > MAX_WORDS or target_words > MAX_WORDS:
        return False, "length"

    ratio = source_words / target_words
    if ratio > MAX_RATIO or ratio < (1.0 / MAX_RATIO):
        return False, "ratio"

    if not has_valid_script(source, source=True):
        return False, "script"

    if not has_valid_script(target, source=False):
        return False, "script"

    return True, "valid"


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessStats]:
    stats = PreprocessStats()
    stats.raw_rows = len(df)

    require_columns(df)
    cleaned = df.copy()
    stats.after_column_selection = len(cleaned)

    cleaned[SOURCE_COLUMN] = cleaned[SOURCE_COLUMN].map(normalize_amharic)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map(normalize_oromo)

    stats.after_normalization = len(cleaned)

    cleaned[SOURCE_COLUMN] = cleaned[SOURCE_COLUMN].astype("string").str.strip()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype("string").str.strip()

    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    before_dropna = len(cleaned)
    empty_mask = cleaned[DATA_COLUMNS].isna().any(axis=1)
    save_removed_csv(cleaned[empty_mask], "removed_empty.csv")

    cleaned = cleaned.dropna(subset=DATA_COLUMNS)
    stats.after_dropna = len(cleaned)
    stats.removed_empty = before_dropna - stats.after_dropna

    if REMOVE_DUPLICATES:
        before_dedup = len(cleaned)
        dup_mask = cleaned.duplicated(subset=DATA_COLUMNS, keep="first")
        save_removed_csv(cleaned[dup_mask], "removed_duplicates.csv")

        cleaned = cleaned.drop_duplicates(subset=DATA_COLUMNS, keep="first")
        stats.after_deduplication = len(cleaned)
        stats.removed_duplicates = before_dedup - stats.after_deduplication
    else:
        stats.after_deduplication = len(cleaned)

    valid_rows = []
    removed_by_category: dict[str, list] = {
        "length": [],
        "ratio": [],
        "characters": [],
        "script": [],
        "noise": [],
    }

    reasons = {"length": 0, "ratio": 0, "characters": 0, "script": 0, "empty": 0}
    noise_counts = {
        "html": 0,
        "url": 0,
        "email": 0,
        "arabic_script": 0,
        "control_chars": 0,
        "replacement_chars": 0,
        "excessive_punctuation": 0,
    }

    for _, row in cleaned.iterrows():
        source = str(row[SOURCE_COLUMN])
        target = str(row[TARGET_COLUMN])

        is_valid, reason = evaluate_row(source, target)

        if is_valid:
            valid_rows.append(row)
            continue

        if reason.startswith("noise:"):
            removed_by_category["noise"].append(row)
            for noise_type in reason.replace("noise:", "").split(","):
                if noise_type in noise_counts:
                    noise_counts[noise_type] += 1
        elif reason in reasons:
            reasons[reason] += 1
            if reason in removed_by_category:
                removed_by_category[reason].append(row)

    save_removed_csv(removed_by_category["length"], "removed_word_limit.csv")
    save_removed_csv(removed_by_category["ratio"], "removed_ratio_filtered.csv")
    save_removed_csv(removed_by_category["characters"], "removed_max_char_limit.csv")
    save_removed_csv(removed_by_category["script"], "removed_invalid_script.csv")
    save_removed_csv(removed_by_category["noise"], "removed_noise.csv")

    if valid_rows:
        cleaned = pd.DataFrame(valid_rows).reset_index(drop=True)
    else:
        cleaned = cleaned.iloc[0:0].copy()

    stats.after_quality_filter = len(cleaned)
    stats.removed_quality = stats.after_deduplication - stats.after_quality_filter

    stats.removed_length = reasons["length"]
    stats.removed_ratio = reasons["ratio"]
    stats.removed_characters = reasons["characters"]
    stats.removed_script = reasons["script"]

    stats.removed_html = noise_counts["html"]
    stats.removed_url = noise_counts["url"]
    stats.removed_email = noise_counts["email"]
    stats.removed_arabic_script = noise_counts["arabic_script"]
    stats.removed_control_chars = noise_counts["control_chars"]
    stats.removed_replacement_chars = noise_counts["replacement_chars"]
    stats.removed_excessive_punctuation = noise_counts["excessive_punctuation"]

    cleaned = cleaned.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    stats.final_rows = len(cleaned)

    return cleaned, stats


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < 10:
        raise ValueError("Dataset is too small to split.")

    train_df, temp_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Calculate final dataset statistics."""
    source_words = df[SOURCE_COLUMN].map(count_words)
    target_words = df[TARGET_COLUMN].map(count_words)
    source_chars = df[SOURCE_COLUMN].astype(str).str.len()
    target_chars = df[TARGET_COLUMN].astype(str).str.len()

    summary = {
        "rows": int(len(df)),
        "source_total_words": int(source_words.sum()),
        "target_total_words": int(target_words.sum()),
        "source_avg_words": round(float(source_words.mean()), 2),
        "target_avg_words": round(float(target_words.mean()), 2),
        "source_min_words": int(source_words.min()),
        "target_min_words": int(target_words.min()),
        "source_max_words": int(source_words.max()),
        "target_max_words": int(target_words.max()),
        "source_avg_chars": round(float(source_chars.mean()), 2),
        "target_avg_chars": round(float(target_chars.mean()), 2),
        "source_max_chars": int(source_chars.max()),
        "target_max_chars": int(target_chars.max()),
        "duplicate_pairs": int(df.duplicated(subset=DATA_COLUMNS).sum()),
    }

    if DOMAIN_COLUMN in df.columns:
        summary["domain_breakdown"] = (
            df[DOMAIN_COLUMN].fillna("Unknown").value_counts().to_dict()
        )

    return summary


def run_pipeline(input_path: Path = INPUT_CSV_FILE) -> None:
    print("\n" + "=" * 70)
    print("AMHARIC ↔ AFAAN OROMO PREPROCESSING")
    print("=" * 70)

    ensure_directories()
    remove_existing_outputs()

    raw_df = load_csv_input(input_path)
    cleaned_df, stats = clean_dataset(raw_df)

    save_csv(cleaned_df, FINAL_CSV_FILE)

    train_df, valid_df, test_df = split_dataset(cleaned_df)

    save_csv(train_df, TRAIN_CSV_FILE)
    save_csv(valid_df, VALID_CSV_FILE)
    save_csv(test_df, TEST_CSV_FILE)

    dataset_summary = analyze_dataset(cleaned_df)

    report = {
        "configuration": {
            "source_column": SOURCE_COLUMN,
            "target_column": TARGET_COLUMN,
            "domain_column": DOMAIN_COLUMN if DOMAIN_COLUMN in raw_df.columns else None,
            "random_state": RANDOM_STATE,
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "max_ratio": MAX_RATIO,
            "max_chars": MAX_CHARS,
            "use_amharic_orthography_map": USE_AMHARIC_ORTHOGRAPHY_MAP,
            "normalize_oromo_punctuation": NORMALIZE_OROMO_PUNCTUATION,
            "remove_duplicates": REMOVE_DUPLICATES,
            "preserve_case": PRESERVE_CASE,
        },
        "stats": asdict(stats),
        "noise_summary": {
            "html": stats.removed_html,
            "url": stats.removed_url,
            "email": stats.removed_email,
            "arabic_script": stats.removed_arabic_script,
            "control_chars": stats.removed_control_chars,
            "replacement_chars": stats.removed_replacement_chars,
            "excessive_punctuation": stats.removed_excessive_punctuation,
        },
        "dataset_summary": dataset_summary,
        "splits": {
            "train": len(train_df),
            "valid": len(valid_df),
            "test": len(test_df),
        },
    }

    save_report(report, REPORT_FILE)

    print("\n" + "=" * 70)
    print("PREPROCESSING RESULTS")
    print("=" * 70)
    print(f"Raw rows                : {stats.raw_rows:,}")
    print(f"Final rows              : {stats.final_rows:,}")
    print(f"Removed empty rows      : {stats.removed_empty:,}")
    print(f"Removed duplicates      : {stats.removed_duplicates:,}")
    print(f"Removed length rows     : {stats.removed_length:,}")
    print(f"Removed ratio rows      : {stats.removed_ratio:,}")
    print(f"Removed char-limit rows : {stats.removed_characters:,}")
    print(f"Removed script rows     : {stats.removed_script:,}")

    print("\n" + "-" * 70)
    print("NOISE DETECTION")
    print("-" * 70)
    print(f"HTML tags               : {stats.removed_html:,}")
    print(f"URLs                    : {stats.removed_url:,}")
    print(f"Email addresses         : {stats.removed_email:,}")
    print(f"Arabic-script noise     : {stats.removed_arabic_script:,}")
    print(f"Control characters      : {stats.removed_control_chars:,}")
    print(f"Replacement characters  : {stats.removed_replacement_chars:,}")
    print(f"Excessive punctuation   : {stats.removed_excessive_punctuation:,}")

    print("\n" + "-" * 70)
    print("FINAL DATASET WORD STATISTICS")
    print("-" * 70)
    print(f"Amharic total words     : {dataset_summary['source_total_words']:,}")
    print(f"Oromo total words       : {dataset_summary['target_total_words']:,}")
    print(f"Amharic avg words       : {dataset_summary['source_avg_words']:.2f}")
    print(f"Oromo avg words         : {dataset_summary['target_avg_words']:.2f}")
    print(f"Amharic min words       : {dataset_summary['source_min_words']}")
    print(f"Oromo min words         : {dataset_summary['target_min_words']}")
    print(f"Amharic max words       : {dataset_summary['source_max_words']}")
    print(f"Oromo max words         : {dataset_summary['target_max_words']}")

    print("\n" + "-" * 70)
    print("DATASET SPLIT")
    print("-" * 70)
    print(f"Train                   : {len(train_df):,}")
    print(f"Validation              : {len(valid_df):,}")
    print(f"Test                    : {len(test_df):,}")

    print("\n" + "-" * 70)
    print("OUTPUT FILES")
    print("-" * 70)
    print(f"Final CSV               : {FINAL_CSV_FILE}")
    print(f"Train CSV               : {TRAIN_CSV_FILE}")
    print(f"Validation CSV          : {VALID_CSV_FILE}")
    print(f"Test CSV                : {TEST_CSV_FILE}")
    print(f"Report                  : {REPORT_FILE}")
    print(f"Removed CSVs Directory : {REMOVED_DIR}")

    print("\n" + "=" * 70)
    print("✓ PREPROCESSING + NOISE ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess Amharic ↔ Afaan Oromo parallel translation data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_CSV_FILE,
        help="Input classified CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.input)


if __name__ == "__main__":
    main()