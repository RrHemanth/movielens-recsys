from recsys.config import load_config
from recsys.data.download import download_movielens


def main():
    cfg = load_config()
    data_dir = download_movielens(cfg["paths"]["raw"], cfg["data"]["dataset"])
    print(f"Data is in: {data_dir}")
    for f in sorted(data_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()