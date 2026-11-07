from recsys.config import load_config
from recsys.data.build_db import build_database


def main():
    cfg = load_config()
    views = build_database(cfg["paths"]["interim"], cfg["paths"]["duckdb"])
    print(f"Created {len(views)} views in {cfg['paths']['duckdb']}:")
    for v in views:
        print(f"  {v}")


if __name__ == "__main__":
    main()