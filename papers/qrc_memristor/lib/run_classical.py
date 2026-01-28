import argparse
import json
import logging
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# Path fix: 3 levels up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from lib.datasets import get_dataset
from lib.classical_models import ClassicalBenchmark

try:
    from lib.datasets import generate_narma
except ImportError:
    logging.warning("Could not import generate_narma from lib.datasets. Ensure the file exists.")


def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "classical_experiment.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )


def save_json(data, filepath):
    def convert(o):
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, torch.Tensor): return o.item() if o.numel() == 1 else o.tolist()
        return str(o)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=convert)


def load_config_file(config_path):
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found.")
        return {}

    with open(config_path, 'r') as f:
        config = json.load(f)

    flat_config = {}
    if 'experiment' in config:
        flat_config.update(config['experiment'])
    if 'data' in config:
        flat_config.update(config['data'])

    for k, v in config.items():
        if k not in ['experiment', 'data'] and not isinstance(v, dict):
            flat_config[k] = v
    return flat_config


def get_clean_config(args):
    config = {
        "task": args.task,
        "model_type": args.model_type,
        "model_class": f"ClassicalBenchmark_{args.model_type}",
        "n_runs": args.n_runs,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "output_dir": args.output_dir,
        "exp_name": args.exp_name
    }
    if args.task == "narma":
        config.update({
            "data_size": args.data_size,
            "washout": args.washout,
            "train_len": args.train_len
        })
    elif args.task == "mackey_glass":
        config.update({
            "data_size": args.data_size,
            "washout": args.washout,
            "train_len": args.train_len,
        })
    elif args.task == "santa_fe":
        config.update({
            "data_size": args.data_size,
            "washout": args.washout,
            "train_len": args.train_len
        })
    return config


def train_and_evaluate(model, u_train, y_train, u_test, y_test, epochs, lr, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        pred = model(u_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(u_test)
        test_mse = torch.mean((pred_test - y_test) ** 2).item()
    return test_mse


def run_classical_task(args):
    logging.info(f"=== Classical Task: {args.task} | Model: {args.model_type} ===")

    all_mses = []

    for i in range(args.n_runs):
        seed = args.seed + i
        torch.manual_seed(seed)
        np.random.seed(seed)

        if args.task == "narma":
            if 'generate_narma' in globals():
                u_tens, y_tens, _ = generate_narma(data_size=args.data_size, seed=seed, device=args.device)
            else:
                u_raw, y_raw = get_dataset("narma", args.data_size)
                u_tens = torch.tensor(u_raw, dtype=torch.float32).to(args.device)
                y_tens = torch.tensor(y_raw, dtype=torch.float32).to(args.device)
            u_tensor = u_tens
            y_tensor = y_tens
        else:
            u_raw, y_raw = get_dataset(args.task, args.data_size)
            if torch.is_tensor(u_raw):
                u_tensor = u_raw.clone().detach().float()
            else:
                u_tensor = torch.tensor(u_raw, dtype=torch.float32)
            if torch.is_tensor(y_raw):
                y_tensor = y_raw.clone().detach().float()
            else:
                y_tensor = torch.tensor(y_raw, dtype=torch.float32)

        if u_tensor.ndim == 2:
            u_tensor = u_tensor.reshape(1, -1, 1).to(args.device)
            y_tensor = y_tensor.reshape(1, -1, 1).to(args.device)
        elif u_tensor.ndim == 3:
            u_tensor = u_tensor.to(args.device)
            y_tensor = y_tensor.to(args.device)

        washout = args.washout
        train_len = args.train_len
        if args.task == "santa_fe" and train_len > u_tensor.shape[1] - 100:
            train_len = int(u_tensor.shape[1] * 0.7)
            logging.warning(f"Adjusted train_len to {train_len} for Santa Fe dataset.")

        u_train = u_tensor[:, washout:train_len, :]
        y_train = y_tensor[:, washout:train_len, :]
        u_test = u_tensor[:, train_len:, :]
        y_test = y_tensor[:, train_len:, :]

        model = ClassicalBenchmark(model_type=args.model_type, input_dim=1)
        mse = train_and_evaluate(model, u_train, y_train, u_test, y_test, args.epochs, args.lr, args.device)

        logging.info(f"Run {i + 1} MSE: {mse:.6f}")
        all_mses.append(mse)

    mean_mse = np.mean(all_mses)
    std_mse = np.std(all_mses)
    logging.info(f"FINAL RESULT >> Mean MSE: {mean_mse:.6f} +/- {std_mse:.6f}")

    return {
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "all_mses": all_mses
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Classical Benchmark Experiments",
        allow_abbrev=False
    )

    parser.add_argument("--config", type=str, help="Path to JSON config file to load defaults from")
    parser.add_argument("--task", choices=["narma", "mackey_glass", "santa_fe"])
    parser.add_argument("--model-type", choices=["L", "Q", "L+M", "Q+M"])
    parser.add_argument("--n-runs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--data-size", type=int)
    parser.add_argument("--washout", type=int)
    parser.add_argument("--train-len", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--exp-name", type=str)

    # CHANGE: parse_known_args
    args, unknown = parser.parse_known_args(argv)

    defaults = {
        "task": "narma",
        "model_type": "L",
        "n_runs": 10,
        "epochs": 200,
        "lr": 0.05,
        "data_size": 1000,
        "washout": 20,
        "train_len": 480,
        "seed": 42,
        "device": "cpu",
        "output_dir": "./results_classical",
        "exp_name": ""
    }

    config_vals = {}
    if args.config:
        print(f"Loading configuration from {args.config}...")
        config_vals = load_config_file(args.config)
        if 'size' in config_vals and 'data_size' not in config_vals:
            config_vals['data_size'] = config_vals.pop('size')

    for key, default_val in defaults.items():
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            pass
        elif key in config_vals:
            setattr(args, key, config_vals[key])
        else:
            setattr(args, key, default_val)

    if not args.task:
        print("Error: The --task argument is required.")
        return 1
    if not args.model_type:
        print("Error: The --model_type argument is required.")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name if args.exp_name else f"{args.task}_{args.model_type}"
    save_path = os.path.join(args.output_dir, f"{exp_name}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    setup_logging(save_path)
    logging.info(f"Experiment initialized at {save_path}")
    if args.config:
        logging.info(f"Loaded config from: {args.config}")

    results = run_classical_task(args)

    save_json(get_clean_config(args), os.path.join(save_path, "config.json"))
    save_json(results, os.path.join(save_path, "metrics.json"))

    logging.info("Classical Experiment Finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
