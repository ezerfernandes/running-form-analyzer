import argparse

from core.config import Config


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Running Analysis using various pose estimation models"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        choices=["local", "server"],
        help=(
            "local = open the laptop webcam and show the analysis window. "
            "server = serve the phone-as-camera streaming app over HTTPS on the LAN."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server-mode bind address (ignored in local mode).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8443,
        help="Server-mode HTTPS port (ignored in local mode).",
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
    if args.mode == "server":
        from server import serve

        serve(config, host=args.host, port=args.port)
    else:
        from core.analyzer import Analyzer

        Analyzer(config).run()


if __name__ == "__main__":
    main()
