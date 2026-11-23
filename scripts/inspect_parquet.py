from pathlib import Path

import pandas as pd

# pyrefly: ignore[missing-import]
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PARQUET_PATH = ROOT / "data" / "hm_ecommerce_products_enriched.parquet"


def main():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"File not found: {PARQUET_PATH}")

    # Read Parquet metadata without loading all rows into memory.
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow

    print("=" * 80)
    print("FILE INFO")
    print("=" * 80)
    print(f"Path: {PARQUET_PATH}")
    print(f"Rows: {metadata.num_rows:,}")
    print(f"Row groups: {metadata.num_row_groups:,}")
    print(f"Columns: {len(schema.names)}")

    print("\n" + "=" * 80)
    print("SCHEMA")
    print("=" * 80)

    for field in schema:
        print(f"{field.name:<40} type={str(field.type):<25} nullable={field.nullable}")

    # Load data to inspect actual values and null counts.
    df = pd.read_parquet(PARQUET_PATH)

    print("\n" + "=" * 80)
    print("NULL COUNTS")
    print("=" * 80)
    print(df.isnull().sum().to_string())

    print("\n" + "=" * 80)
    print("SAMPLE ROWS")
    print("=" * 80)
    print(df.head(5).to_string(index=False, max_colwidth=80))

    print("\n" + "=" * 80)
    print("DISTINCT VALUES FOR CATEGORICAL COLUMNS")
    print("=" * 80)

    categorical_candidates = [
        "product_group_name",
        "product_type_name",
        "colour_group_name",
        "graphical_appearance_name",
        "index_group_name",
    ]

    for column in categorical_candidates:
        if column not in df.columns:
            continue

        print(f"\n[{column}]")
        print(f"Unique values: {df[column].nunique(dropna=True):,}")
        print(df[column].value_counts(dropna=False).head(30).to_string())

    # Export schema and a small sample for later review.
    schema_path = ROOT / "data" / "parquet_schema.txt"
    sample_path = ROOT / "data" / "parquet_sample.csv"

    schema_path.parent.mkdir(parents=True, exist_ok=True)

    schema_path.write_text(
        str(schema) + "\n",
        encoding="utf-8",
    )
    df.head(100).to_csv(
        sample_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Schema: {schema_path}")
    print(f"Sample: {sample_path}")


if __name__ == "__main__":
    main()
