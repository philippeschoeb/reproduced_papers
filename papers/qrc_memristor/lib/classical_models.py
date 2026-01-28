import torch
import torch.nn as nn


class ClassicalBenchmark(nn.Module):
    """
    Implements the 4 specific classical models defined by the user.
    Input 'u' corresponds to the variable 'y_t' in the user's provided formulas.

    Models:
    1. L (Linear):
       Out = a*u_t + b

    2. Q (Cubic Polynomial):
       Out = a*u_t^3 + b*u_t^2 + c*u_t + d

    3. L+M (Linear + Memory):
       Out = a*u_t + b*u_{t-1} + c

    4. Q+M (Cubic Polynomial + Memory):
       Full 3rd-degree polynomial in two variables (u_t, u_{t-1}).
       Contains terms: u_t^3, u_t^2*u_{t-1}, ..., u_{t-1}^3 is NOT in your list,
       but typically included in full Poly3. I strictly follow your list below:
       Terms: u_t^3, u_t^2*u_{t-1}, u_t*u_{t-1}^2, u_t^2, u_t*u_{t-1}, u_{t-1}^2, u_t, u_{t-1}, bias
    """

    def __init__(self, model_type="L", input_dim=1):
        super().__init__()
        self.model_type = model_type
        self.input_dim = input_dim

        # Define the input dimension for the final linear layer
        if model_type == "L":
            # Features: u_t (1 term)
            self.net = nn.Linear(input_dim, 1)

        elif model_type == "Q":
            # Features: u_t^3, u_t^2, u_t (3 terms)
            self.net = nn.Linear(input_dim * 3, 1)

        elif model_type == "L+M":
            # Features: u_t, u_{t-1} (2 terms)
            self.net = nn.Linear(input_dim * 2, 1)

        elif model_type == "Q+M":
            # Features from your formula:
            # 1. u_t^3
            # 2. u_t^2 * u_{t-1}
            # 3. u_t * u_{t-1}^2
            # 4. u_t^2
            # 5. u_t * u_{t-1}
            # 6. u_{t-1}^2
            # 7. u_t
            # 8. u_{t-1}
            # Total = 8 weights + 1 bias
            self.net = nn.Linear(input_dim * 8, 1)

        else:
            raise ValueError(f"Unknown classical model type: {model_type}")

    def forward(self, u_seq):
        """
        Args:
            u_seq: Input sequence of shape (Batch, Seq_Len, 1)
        Returns:
            Output sequence of shape (Batch, Seq_Len, 1)
        """
        batch_size, seq_len, _ = u_seq.shape
        device = u_seq.device

        # 1. Prepare u_t
        u_t = u_seq  # (B, T, 1)

        # 2. Prepare u_{t-1} (Shifted, padded with 0 at t=0)
        u_tm1 = torch.cat([torch.zeros(batch_size, 1, 1, device=device), u_seq[:, :-1, :]], dim=1)

        # 3. Construct Feature Vector based on Model Type
        if self.model_type == "L":
            features = u_t

        elif self.model_type == "Q":
            features = torch.cat([
                u_t ** 3,
                u_t ** 2,
                u_t
            ], dim=2)

        elif self.model_type == "L+M":
            features = torch.cat([
                u_t,
                u_tm1
            ], dim=2)

        elif self.model_type == "Q+M":
            features = torch.cat([
                u_t ** 3,
                (u_t ** 2) * u_tm1,
                u_t * (u_tm1 ** 2),
                u_t ** 2,
                u_t * u_tm1,
                u_tm1 ** 2,
                u_t,
                u_tm1
            ], dim=2)

        # 4. Apply Linear Layer (Weights + Bias)
        return self.net(features)
