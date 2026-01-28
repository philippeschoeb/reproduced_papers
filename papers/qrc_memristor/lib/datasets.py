import numpy as np
import torch
import os


# --- 1. NARMA (Time Series) ---
def generate_narma(
    data_size: int = 1000,
    x_low: float = 0.0, x_high: float = 0.5,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """
    Paper NARMA variant:
      x_t ~ Uniform(0, 0.5)
      y_{t+1} = 0.4 y_t + 0.4 y_t y_{t-1} + 0.6 x_t^3 + 0.1

    Returns:
      x: (N,1) inputs x_t
      y: (N,1) targets y_{t+1} aligned with x_t
      y_full: (N+1,) y_0..y_N
    """
    if seed is not None:
        g = torch.Generator(device=str(device))
        g.manual_seed(seed)
    else:
        g = None

    x = torch.empty(data_size, device=device, dtype=dtype).uniform_(x_low, x_high, generator=g)

    y_full = torch.empty(data_size + 1, device=device, dtype=dtype)
    y_full[0] = 0.0  # y_0
    y_prev = torch.tensor(0.0, device=device, dtype=dtype)  # y_{-1} assumed 0
    y_curr = torch.tensor(0.0, device=device, dtype=dtype)  # y_0

    for t in range(data_size):
        y_next = 0.4 * y_curr + 0.4 * y_curr * y_prev + 0.6 * (x[t] ** 3) + 0.1
        y_full[t + 1] = y_next
        y_prev, y_curr = y_curr, y_next

    # x_t -> y_{t+1}
    return x.unsqueeze(1), y_full[1:].unsqueeze(1), y_full


# --- 2. MACKEY-GLASS (Chaotic ODE) ---
def generate_mackey_glass(n_samples, tau=17, beta=0.2, gamma=0.1, n=10,
                          integration_step=0.1, sampling_step=1.0):
    """
    Generates Mackey-Glass using RK4 integration.
    """
    steps_per_sample = int(sampling_step / integration_step)
    delay_steps = int(tau / integration_step)

    burn_in_samples = 200
    total_samples = n_samples + burn_in_samples
    total_integration_steps = total_samples * steps_per_sample

    buffer_size = total_integration_steps + delay_steps
    x_history = np.zeros(buffer_size)
    x_history[:delay_steps] = 0.5 + 0.05 * np.random.rand(delay_steps)

    for i in range(delay_steps, buffer_size - 1):
        x_val = x_history[i]
        x_tau = x_history[i - delay_steps]

        def mg_deriv(val, val_tau):
            return (beta * val_tau) / (1.0 + val_tau ** n) - gamma * val

        k1 = mg_deriv(x_val, x_tau)
        k2 = mg_deriv(x_val + 0.5 * integration_step * k1, x_tau)
        k3 = mg_deriv(x_val + 0.5 * integration_step * k2, x_tau)
        k4 = mg_deriv(x_val + integration_step * k3, x_tau)

        x_history[i + 1] = x_val + (integration_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    generated_data = x_history[delay_steps::steps_per_sample]
    final_data = generated_data[burn_in_samples: burn_in_samples + n_samples + 1]

    inputs = final_data[:-1]
    targets = final_data[1:]

    return inputs, targets


# --- 3. SANTA FE LASER (Real Data - Local File) ---
def load_santa_fe(n_samples=1000, data_dir="./data"):
    """
    Loads the Santa Fe Laser Dataset from a local file.
    Expected location: ./data/santafe_laser.txt
    """
    filename = "santafe_laser.txt"
    filepath = os.path.join(data_dir, filename)

    # 1. Check if file exists
    if not os.path.exists(filepath):
        # Try looking one level up just in case script is run from inside src/ or scripts/
        filepath_up = os.path.join("..", data_dir, filename)
        if os.path.exists(filepath_up):
            filepath = filepath_up
        else:
            raise FileNotFoundError(
                f"Could not find the dataset at: {os.path.abspath(filepath)}\n"
                f"Please ensure 'santafe_laser.txt' is inside the '{data_dir}' folder."
            )

    # 2. Load Data
    try:
        # genfromtxt handles standard text files with numbers separated by newlines
        data = np.genfromtxt(filepath)
    except Exception as e:
        raise ValueError(f"Error reading {filepath}. Ensure it contains only numbers.") from e

    # 3. Normalize to [0, 1]
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min == 0:
        print("Warning: Dataset is constant. Normalization skipped.")
    else:
        data = (data - data_min) / (data_max - data_min)

    # 4. Truncate to requested size
    # We need n_samples input/target pairs, so we need n_samples + 1 data points
    if n_samples > len(data) - 1:
        print(f"Requested {n_samples} samples, but file has {len(data)}. Using max available.")
        n_samples = len(data) - 1

    inputs = data[:n_samples]
    targets = data[1:n_samples + 1]

    return inputs, targets


# --- UNIFIED LOADER ---
def get_dataset(task, n_samples=1000, **kwargs):
    if task == "narma":
        u, y, _ = generate_narma(n_samples)
        return u, y
    elif task == "mackey_glass":
        return generate_mackey_glass(n_samples, **kwargs)
    elif task == "santa_fe":
        return load_santa_fe(n_samples, **kwargs)
    elif task == "nonlinear":
        x = torch.linspace(0, 1, 85).numpy()
        y = x ** 4
        return x, y
    else:
        raise ValueError(f"Unknown task: {task}")
