"""
Quantum Adversarial Machine Learning - Library
===============================================

Implementation of "Quantum Adversarial Machine Learning"
by Lu et al. (2020).

This reproduction uses MerLin (photonic) as the primary backend,
implementing variational quantum circuits with beam splitter meshes
and phase shifter encoding.

Modules:
    - circuits: Variational quantum circuit classifiers (MerLin)
    - models: Quantum classifier architectures
    - datasets: Data loading (MNIST, Ising model)
    - attacks: Adversarial attack methods (FGSM, BIM, PGD, MIM)
    - defense: Adversarial training and defense strategies
    - training: Training loops and optimization
    - visualization: Plotting utilities
    - runner: Main entry point for experiments
"""

from .circuits import (
    MerLinQuantumClassifier,
    MerLinAmplitudeClassifier,
    MerLinAmplitudeClassifierDirect,
    create_classifier_circuit,
    ScaleLayer,
    PENNYLANE_AVAILABLE,
)

# Conditionally import PennyLane classes
if PENNYLANE_AVAILABLE:
    from .circuits import (
        PennyLaneQuantumClassifier,
        PennyLaneHybridClassifier,
        PennyLaneSimpleClassifier,
    )

from .datasets import (
    MNISTBinaryDataset,
    MNISTMulticlassDataset,
    IsingDataset,
    QAHDataset,
    TopologicalPhaseDataset,
    SpiralDataset,
    create_dataloaders,
    amplitude_encode,
)
from .models import (
    QuantumClassifier,
    HybridQuantumClassifier,
    ClassicalBaseline,
    ClassicalCNN,
    ClassicalFNN,
    create_model,
)
from .attacks import (
    fgsm_attack,
    bim_attack,
    pgd_attack,
    mim_attack,
    functional_attack,
    functional_fgsm_attack,
    transfer_attack,
    run_transfer_attack_experiment,
    generate_adversarial_examples,
    compute_fidelity,
    evaluate_attack_success,
    add_random_noise,
    evaluate_with_photon_loss,
    compare_noise_vs_adversarial,
    run_noise_comparison_experiment,
)
from .defense import (
    adversarial_training_step,
    AdversarialTrainer,
    evaluate_robustness,
)
from .training import (
    train_epoch,
    evaluate,
    Trainer,
    train_model,
)
from .runner import main, set_seed

__all__ = [
    # Circuits - Photonic
    "MerLinQuantumClassifier",
    "MerLinAmplitudeClassifier",
    "MerLinAmplitudeClassifierDirect",
    "create_classifier_circuit",
    "ScaleLayer",
    "PENNYLANE_AVAILABLE",
    # Circuits - Gate-based (conditional)
    "PennyLaneQuantumClassifier",
    "PennyLaneHybridClassifier",
    "PennyLaneSimpleClassifier",
    # Models
    "QuantumClassifier",
    "HybridQuantumClassifier",
    "ClassicalBaseline",
    "ClassicalCNN",
    "ClassicalFNN",
    "create_model",
    # Datasets
    "MNISTBinaryDataset",
    "MNISTMulticlassDataset",
    "IsingDataset",
    "QAHDataset",
    "TopologicalPhaseDataset",
    "SpiralDataset",
    "create_dataloaders",
    "amplitude_encode",
    # Attacks
    "fgsm_attack",
    "bim_attack",
    "pgd_attack",
    "mim_attack",
    "functional_attack",
    "functional_fgsm_attack",
    "transfer_attack",
    "run_transfer_attack_experiment",
    "generate_adversarial_examples",
    "compute_fidelity",
    "evaluate_attack_success",
    "add_random_noise",
    "evaluate_with_photon_loss",
    "compare_noise_vs_adversarial",
    "run_noise_comparison_experiment",
    # Defense
    "adversarial_training_step",
    "AdversarialTrainer",
    "evaluate_robustness",
    # Training
    "train_epoch",
    "evaluate",
    "Trainer",
    "train_model",
    # Runner
    "main",
    "set_seed",
]

__version__ = "1.0.0"
