import sys
import argparse
from .run_quantum import main as main_quantum
from .run_classical import main as main_classical


# CHANGE: Function name must be 'train_and_evaluate' matching runtime_lib expectations
# CHANGE: Accepts 2 arguments from the runtime
def train_and_evaluate(runtime_cfg, run_dir):
    # We grab the raw command line arguments because runtime_cfg
    # might be empty if cli.json is empty.
    raw_argv = sys.argv[1:]

    # 1. Check for --mode flag to decide which script to run
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', choices=['quantum', 'classical'], default='quantum')

    # parse_known_args lets us see --mode without crashing on other flags
    args, _ = parser.parse_known_args(raw_argv)

    # 2. Dispatch to the correct script
    if args.mode == 'classical':
        return main_classical(raw_argv)
    else:
        return main_quantum(raw_argv)


if __name__ == "__main__":
    # Allow testing this file directly (mocking the runtime call)
    train_and_evaluate(None, ".")
