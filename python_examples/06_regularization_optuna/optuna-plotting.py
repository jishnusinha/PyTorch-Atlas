"""Requires optional Optuna visualization extras."""
import torch

def show_plots(study):
    try:
        import optuna
        from optuna.visualization import (
            plot_optimization_history,
            plot_parallel_coordinate,
            plot_param_importances,
        )
    except ImportError as error:
        raise RuntimeError("Install Optuna visualization extras to create plots") from error
    figures = [
        plot_optimization_history(study),
        plot_parallel_coordinate(study),
        plot_param_importances(study),
    ]
    assert all(figure is not None for figure in figures)
    for figure in figures:
        figure.show()  # Local interactive output; no hosted tracking service is required.

print("Call show_plots(study) after optimization when Optuna visualization extras are available.")
assert callable(show_plots)
