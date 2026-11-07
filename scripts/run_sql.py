import sys
from pathlib import Path

import duckdb

from recsys.config import load_config


def main():
    cfg = load_config()
    con = duckdb.connect(str(cfg["paths"]["duckdb"]), read_only=True)
    for path in sys.argv[1:]:
        print(f"\n=== {path} ===")
        print(con.sql(Path(path).read_text()))
    con.close()


if __name__ == "__main__":
    main()