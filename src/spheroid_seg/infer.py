"""Inference script stub."""

import argparse


def main() -> None:
    """Parse CLI arguments and exit with a not-implemented message."""
    parser = argparse.ArgumentParser(
        description="Run inference with the spheroid segmentation model.",
    )
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    print(f"Inference not implemented (config: {args.config}).")


if __name__ == "__main__":
    main()
