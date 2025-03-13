from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import ndarray
from torch import Tensor

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence
    from numpy.typing import NDArray

plt.rcParams["figure.figsize"] = (16,9)

def draw_images(images: Sequence[Tensor | NDArray] | Tensor | NDArray | Any):
    if not isinstance(images, (list, tuple)):
        images = [images]
    _, axs = plt.subplots(nrows=1, ncols=len(images), squeeze=False)
    for idx, img_data in enumerate(images):
        if isinstance(img_data, ndarray):
            img_data = Tensor(img_data)
        if img_data.dim() == 2:
            img_data = img_data.unsqueeze(0)
        img_data = img_data.permute(1, 2, 0)
        axs[0, idx].imshow(img_data)

    plt.tight_layout()
    plt.show()

