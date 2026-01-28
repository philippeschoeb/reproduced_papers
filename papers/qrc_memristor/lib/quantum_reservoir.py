import merlin as ml
import perceval as pcvl
import perceval.components as comp
from lib.feedback import FeedbackLayer, FeedbackLayerNARMA
from lib.training import *


class QuantumReservoirFeedback(torch.nn.Module):
    def __init__(self, input_dim=1, n_modes=3, memory=5):
        super().__init__()

        self.n_modes = n_modes

        # Create circuit that accepts TWO inputs: px_0 (data) and px_1 (feedback)
        circuit = self.gen_circuit(n_modes)

        self.quantum_layer = ml.QuantumLayer(
            input_size=input_dim + 1,
            circuit=circuit,
            trainable_parameters=["theta"],
            input_parameters=["px"],
            input_state=[0, 1, 0],
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            no_bunching=True
        )

        self.feedback = FeedbackLayer(memory_size=memory)

    def gen_circuit(self, n_modes):
        """Generate quantum circuit with feedback input"""
        # Encoding with input data (px_0)
        phi_enc = pcvl.P(f"px_{0}")
        u_enc_u_1 = (pcvl.Circuit(2, name="U_enc, U_1")
                     .add((0, 1), comp.BS())
                     .add(1, comp.PS(phi=phi_enc))
                     .add((0, 1), comp.BS())
                     .add(1, comp.PS(phi=pcvl.P(f"theta_{1}"))))

        # Memory/feedback circuit with feedback input (px_1)
        u_mem = (pcvl.Circuit(2, name="U_mem")
                 .add((0, 1), comp.BS())
                 .add(0, comp.PS(phi=pcvl.P(f"px_{1}")))  # FEEDBACK INPUT
                 .add((0, 1), comp.BS()))

        # Measurement circuit
        u_2 = (pcvl.Circuit(2, name="U_2")
               .add(1, comp.PS(phi=pcvl.P(f"theta_{2}")))
               .add((0, 1), comp.BS())
               .add(1, comp.PS(phi=pcvl.P(f"theta_{3}")))
               .add((0, 1), comp.BS()))

        quantum_circ = pcvl.Circuit(n_modes)
        quantum_circ.add(0, u_enc_u_1)
        quantum_circ.add(1, u_mem)
        quantum_circ.add(0, u_2)

        return quantum_circ

    def forward(self, x):
        batch_size = x.shape[0]

        # If we have no R yet, start from R=0.5 (balanced) like typical initialization
        if self.feedback.last_feedback is None:
            R_t = torch.full((batch_size, 1), 0.5, device=x.device, dtype=x.dtype)
        else:
            R_scalar = self.feedback.last_feedback
            R_t = R_scalar.view(1, 1).expand(batch_size, 1)

        # compute the encoding phase
        phi_enc = encode_phase(x)

        # compute theta from R_t
        theta_t = R_to_theta(R_t)

        # pass phi_enc and theta_t into the quantum layer as px_0 and px_1
        quantum_input = torch.cat([phi_enc, theta_t], dim=1)
        out = self.quantum_layer(quantum_input)

        # update R for next step (using current output)
        _ = self.feedback(out)

        return out

    def reset_feedback(self):
        """Call this at the start of each sequence/epoch"""
        self.feedback.reset()


class QuantumReservoirFeedbackTimeSeries(torch.nn.Module):
    def __init__(self, input_dim=1, n_modes=3, memory=5):
        super().__init__()
        self.n_modes = n_modes
        self.circuit = self.gen_circuit(n_modes)

        self.quantum_layer = ml.QuantumLayer(
            input_size=input_dim + 1,  # input + feedback
            circuit=self.circuit,
            trainable_parameters=[],  # Fixed reservoir
            input_parameters=["px_0", "px_1"],
            input_state=[0, 1, 0],
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            no_bunching=True
        )

        self.feedback = FeedbackLayerNARMA(memory_size=memory)

    def gen_circuit(self, n_modes):
        # Encoding with input data (px_0)
        phi_enc = pcvl.P(f"px_{0}")
        u_enc = (pcvl.Circuit(2, name="U_enc")
                 .add((0, 1), comp.BS())
                 .add(1, comp.PS(phi=phi_enc))
                 .add((0, 1), comp.BS()))

        # Memory/feedback circuit with feedback input (px_1)
        u_mem = (pcvl.Circuit(2, name="U_mem")
                 .add((0, 1), comp.BS())
                 .add(0, comp.PS(phi=pcvl.P(f"px_{1}")))
                 .add((0, 1), comp.BS()))

        quantum_circ = pcvl.Circuit(n_modes)
        quantum_circ.add(0, u_enc)
        quantum_circ.add(1, u_mem)

        return quantum_circ

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, input_dim)
        or (sequence_length, input_dim) if batch_size=1
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, seq_len, _ = x.shape
        outputs = []

        # Reset feedback at the start of the sequence
        self.reset_feedback()

        for t in range(seq_len):
            # 1. Get current input step
            x_t = x[:, t, :]  # (Batch, 1)

            # 2. Get current feedback state R_t
            R_scalar = self.feedback.get_R(device=x.device, dtype=x.dtype)

            # If batch > 1, ensure R_scalar matches batch dim.
            if R_scalar.dim() == 0:
                R_t = R_scalar.view(1, 1).expand(batch_size, 1)
            else:
                R_t = R_scalar.view(batch_size, 1)

            # 3. Encode Inputs
            phi_enc = encode_phase(x_t)
            theta_t = R_to_theta(R_t)

            # 4. Step the Quantum Layer
            step_input = torch.cat([phi_enc, theta_t], dim=1)
            out_t = self.quantum_layer(step_input)

            # 5. Update Feedback for next step
            _ = self.feedback(out_t)

            outputs.append(out_t)

        return torch.stack(outputs, dim=1)

    def reset_feedback(self):
        self.feedback.reset()


class QuantumReservoirNoMem(torch.nn.Module):
    """
    Quantum reservoir baseline with:
    - input: x -> px_0
    - no feedback loop
    - memristor phase is a trainable parameter (phi_mem)
    """
    def __init__(self, input_dim=1, n_modes=3):
        super().__init__()
        self.n_modes = n_modes

        circuit = self.gen_circuit(n_modes)

        self.quantum_layer = ml.QuantumLayer(
            input_size=input_dim,
            circuit=circuit,
            trainable_parameters=["theta"],
            input_parameters=["px"],
            input_state=[0, 1, 0],
            measurement_strategy=ml.MeasurementStrategy.PROBABILITIES,
            no_bunching=True
        )

    def gen_circuit(self, n_modes):
        # Encoding with input data (px_0)
        phi_enc = pcvl.P("px_0")
        u_enc_u_1 = (pcvl.Circuit(2, name="U_enc, U_1")
                     .add((0, 1), comp.BS())
                     .add(1, comp.PS(phi=phi_enc))
                     .add((0, 1), comp.BS())
                     .add(1, comp.PS(phi=pcvl.P("theta_1"))))

        # Trainable angle instead of memristor
        u_mem = (pcvl.Circuit(2, name="U_mem_static")
                 .add((0, 1), comp.BS())
                 # .add(0, comp.PS(phi=pcvl.P("theta_2")))
                 .add(0, comp.PS(phi=np.pi/2))
                 .add((0, 1), comp.BS()))

        # Measurement circuit
        u_2 = (pcvl.Circuit(2, name="U_2")
               .add(1, comp.PS(phi=pcvl.P("theta_3")))
               .add((0, 1), comp.BS())
               .add(1, comp.PS(phi=pcvl.P("theta_4")))
               .add((0, 1), comp.BS()))

        quantum_circ = pcvl.Circuit(n_modes)
        quantum_circ.add(0, u_enc_u_1)
        quantum_circ.add(1, u_mem)
        quantum_circ.add(0, u_2)

        return quantum_circ

    def forward(self, x):
        # x shape: (batch, 1)
        return self.quantum_layer(x)

