import pandas as pd
import re
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = Path("../data/final.csv")

SOURCE_COLUMN = "Amharic"
TARGET_COLUMN = "Oromo"


# ============================================================
# WORD COUNT FUNCTION
# ============================================================

def count_words(text):
    """
    Count actual words in a sentence.

    Supports:
    - Amharic (Ge'ez) characters
    - Oromo/Latin characters
    - Numbers

    Punctuation and whitespace are not counted as words.
    """

    if pd.isna(text):
        return 0

    text = str(text)

    # Amharic/Ge'ez Unicode range: U+1200–U+137F
    # Latin letters: A-Z / a-z
    # Numbers are also allowed inside words.
    pattern = r"[A-Za-z0-9\u1200-\u137F]+"

    return len(re.findall(pattern, text))


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    """Load and validate the final translation dataset."""

    if not FILE_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {FILE_PATH}"
        )

    df = pd.read_csv(FILE_PATH, encoding="utf-8")

    required_columns = [SOURCE_COLUMN, TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    return df


# ============================================================
# CALCULATE STATISTICS
# ============================================================

def calculate_statistics(df, column):
    """Calculate word statistics for one language."""

    word_counts = df[column].apply(count_words)

    return {
        "total_words": int(word_counts.sum()),
        "average_words": word_counts.mean(),
        "minimum_words": int(word_counts.min()),
        "maximum_words": int(word_counts.max()),
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_statistics(language, statistics):
    """Display language statistics in a clear format."""

    print(f"\n{'=' * 60}")
    print(f"{language.upper()} WORD STATISTICS")
    print(f"{'=' * 60}")

    print(
        f"Total words              : "
        f"{statistics['total_words']:,}"
    )

    print(
        f"Average words/sentence   : "
        f"{statistics['average_words']:.2f}"
    )

    print(
        f"Minimum words/sentence   : "
        f"{statistics['minimum_words']}"
    )

    print(
        f"Maximum words/sentence   : "
        f"{statistics['maximum_words']}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("AMHARIC ↔ AFAAN OROMO DATASET WORD ANALYSIS")
    print("=" * 60)

    # Load final dataset
    df = load_dataset()

    print(f"\nTotal sentence pairs : {len(df):,}")

    # Calculate statistics
    amharic_stats = calculate_statistics(
        df,
        SOURCE_COLUMN
    )

    oromo_stats = calculate_statistics(
        df,
        TARGET_COLUMN
    )

    # Display results
    display_statistics(
        "Amharic",
        amharic_stats
    )

    display_statistics(
        "Afaan Oromo",
        oromo_stats
    )

    print("\n" + "=" * 60)
    print("WORD ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()