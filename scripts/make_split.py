from recsys.config import load_config
from recsys.data.split import time_split


def main():
    cfg = load_config()
    counts = time_split(
        ratings_path=cfg["paths"]["interim"] / "ratings.parquet",
        out_dir=cfg["paths"]["processed"],
        val_start=cfg["split"]["val_start"],
        test_start=cfg["split"]["test_start"],
    )
    for name, n in counts.items():
        print(f"  {name:6s} {n:>12,}")
    print(f"  {'total':6s} {sum(counts.values()):>12,}")


if __name__ == "__main__":
    main()