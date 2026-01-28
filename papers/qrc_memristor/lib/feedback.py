from collections import deque
import torch


class FeedbackLayer(torch.nn.Module):
    def __init__(self, memory_size: int):
        super().__init__()
        self.memory_size = memory_size
        self.a = torch.nn.Parameter(torch.randn(1))
        self.b = torch.nn.Parameter(torch.randn(1))
        self.mem = deque(maxlen=memory_size)
        self.last_feedback = None

    def reset(self):
        self.mem.clear()
        self.last_feedback = None

    def forward(self, p):
        """
        Args:
            p: quantum output probabilities, shape (batch, dim)
        Returns:
            r_t: updated reflectivity (scalar tensor)
        """
        p_mem = p[:, 2].mean()

        # update memory
        self.mem.append(p_mem.detach())
        mem_tensor = torch.stack(list(self.mem))

        r_raw = (self.a * mem_tensor + self.b).mean()
        r_t = torch.sigmoid(r_raw)

        self.last_feedback = r_t   # this is R_t now

        return r_t


class FeedbackLayerNARMA(torch.nn.Module):
    def __init__(self, memory_size: int, r_0: float = 0.5, update_index: int = 1, eps: float = 1e-6):
        super().__init__()
        self.update_index = update_index
        self.r_0 = float(r_0)
        self.eps = eps
        self.register_buffer("memory_size", torch.tensor(float(memory_size)))
        self.R = None  # Internal state storage

    def reset(self):
        self.R = None

    def get_R(self, device=None, dtype=None):
        """Helper to get current R, initializing if necessary."""
        if self.R is None:
            return torch.tensor(self.r_0, device=device, dtype=dtype)
        return self.R

    def forward(self, p):
        """
        Updates the internal memristor state based on input probability p.
        Args:
            p: quantum output probabilities (Batch, Dim)
        Returns:
            R_next: The NEW reflectivity value (Scalar or Batch)
        """
        # 1. Extract the probability driving the memory (usually index 0 or 2)
        p_update = p[:, self.update_index].mean()

        if self.update_index == 2:
            p_update = 1.0 - p_update

        # 2. Get previous R
        R_prev = self.get_R(device=p_update.device, dtype=p_update.dtype)

        # 3. NARMA Update Equation: R(t) = R(t-1) + (P(t) - R(t-1)) / tau
        R_next = R_prev + (p_update - R_prev) / self.memory_size
        R_next = torch.clamp(R_next, self.eps, 1.0 - self.eps)

        # 4. Save state
        self.R = R_next.detach()

        return R_next
