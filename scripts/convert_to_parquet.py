import time

from recsys.config import load_config
from recsys.data.to_parquet import convert_all


def main():
    cfg = load_config()
    raw_dir = cfg["paths"]["raw"] / cfg["data"]["dataset"]
    out_dir = cfg["paths"]["interim"]

    start = time.time()
    row_counts = convert_all(raw_dir, out_dir)
    print(f"\nDone in {time.time() - start:.1f} seconds. Row counts:")
    for name, n in row_counts.items():
        print(f"  {name:15s} {n:>12,}")


if __name__ == "__main__":
    main()