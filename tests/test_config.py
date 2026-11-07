from recsys.config import load_config


def test_config_loads():
    cfg = load_config()

    # The seed we set in configs/base.yaml should come through unchanged.
    assert cfg["seed"] == 42

    # Every path should be absolute, so code works no matter where it's run from.
    for name, path in cfg["paths"].items():
        assert path.is_absolute(), f"{name} is not an absolute path"