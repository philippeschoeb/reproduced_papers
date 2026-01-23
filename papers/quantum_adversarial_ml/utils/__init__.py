"""
Visualization Utilities for Quantum Adversarial ML
===================================================

Post-hoc figure generation from saved experiment results.

Usage:
    python -m utils.generate_figures results/train_quantum/run_20240120-123456
    python -m utils.generate_figures results/ --all
"""

from .generate_figures import generate_all_figures, generate_figures_for_run
from .plot_attacks import plot_attack_from_results
from .plot_comparison import plot_noise_comparison_from_results
from .plot_training import plot_training_from_results

__all__ = [
    "generate_all_figures",
    "generate_figures_for_run",
    "plot_training_from_results",
    "plot_attack_from_results",
    "plot_noise_comparison_from_results",
]
