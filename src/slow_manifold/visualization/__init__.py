"""Visualization helpers that consume structured results."""

from .task import plot_task_batch
from .training_dynamics import (
    create_latent_vector_field_movie,
    plot_latent_jacobian_snapshot,
    plot_latent_vector_field_snapshot,
    plot_representative_outputs,
    plot_training_curves,
    render_latent_dynamics_collection,
    render_latent_jacobian_collection,
    render_latent_vector_field_collection,
)

__all__ = [
    "create_latent_vector_field_movie",
    "plot_latent_jacobian_snapshot",
    "plot_latent_vector_field_snapshot",
    "plot_representative_outputs",
    "plot_task_batch",
    "plot_training_curves",
    "render_latent_dynamics_collection",
    "render_latent_jacobian_collection",
    "render_latent_vector_field_collection",
]
