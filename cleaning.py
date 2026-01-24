import csv
import os

# =========================
# Configuration
# =========================
INPUT_CSV = (
    "/Users/amelielaura/Documents/Project6/outputs/eval_only_motif_based/evaluation/"
    "eval_motif_only__ALTmustHaveDeletions__A100_B100_both100.csv"
)
OUTPUT_CSV = (
    "/Users/amelielaura/Documents/Project6/outputs/eval_only_motif_based/evaluation/"
    "eval_motif_only__ALTmustHaveDeletions__A100_B100_both100_cleaned.csv"
)

def fill_missing_value(value):
    """
    Replace missing or empty values with 'NA' for better consistency.
    """
    if value is None or str(value).strip() == "":
        return "NA"
    return value

def clean_csv_file(input_path, output_path):
    """
    Reads a CSV file, replaces missing values with 'NA', and writes the cleaned data
    to a new CSV file in the same format.
    """
    print(f"Reading data from: {input_path}")
    with open(input_path, newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames

        cleaned_rows = []
        for row in reader:
            cleaned_row = {key: fill_missing_value(row.get(key)) for key in fieldnames}
            cleaned_rows.append(cleaned_row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing cleaned data to: {output_path}")
    with open(output_path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print("Finito.")

if __name__ == "__main__":
    clean_csv_file(INPUT_CSV, OUTPUT_CSV)
