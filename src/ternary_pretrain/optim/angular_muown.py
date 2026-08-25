from __future__ import annotations

from torch import Tensor

from ternary_pretrain.optim.muown import Muown
from ternary_pretrain.optim.polar import tangent_projection


class AngularMuown(Muown):
    """Use row-scale-aware tangent gradients before the Muown retraction."""

    def _direction_gradient(
        self, direction: Tensor, row_magnitude: Tensor, gradient: Tensor
    ) -> Tensor:
        return row_magnitude * tangent_projection(direction, gradient)
