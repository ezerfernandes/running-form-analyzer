import argparse
from core.config import Config
from core.analyzer import Analyzer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Running Analysis using various pose estimation models"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="blazepose",
        choices=["movenet", "blazepose"],
        help="Type of pose estimation model to use",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="right",
        choices=["left", "right"],
        help="Side of the body to analyze",
    )
    parser.add_argument(
        "--runner_height", type=float, default=182, help="Height of the runner in cm"
    )
    parser.add_argument(
        "--sex",
        type=str,
        default="male",
        choices=["male", "female"],
        help="Runner sex; selects the torso-length anthropometric ratio.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = Config.from_args(args)
    analyzer = Analyzer(config)
    analyzer.run()


if __name__ == "__main__":
    main()
