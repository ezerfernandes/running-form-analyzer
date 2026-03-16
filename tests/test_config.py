from argparse import Namespace

from core.config import Config, EDGES, HFOV_DEG, IMAGE_WIDTH_PX


class TestConfig:
    def test_from_args(self):
        args = Namespace(side="left", model_type="blazepose", runner_height=182)
        config = Config.from_args(args)
        assert config.side == "left"
        assert config.model_type == "blazepose"
        assert config.runner_height == 182

    def test_to_dict(self):
        config = Config(side="right", model_type="movenet", runner_height=175)
        d = config.to_dict()
        assert d == {"side": "right", "model_type": "movenet", "runner_height": 175}

    def test_constants(self):
        assert HFOV_DEG == 74
        assert IMAGE_WIDTH_PX == 1280
        assert (0, 1) in EDGES
        assert len(EDGES) == 18
