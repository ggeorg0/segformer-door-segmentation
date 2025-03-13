from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import Transform
from transformers import SegformerImageProcessor


class DoorsDataset(Dataset):
    """Load 3000 images dataset of doors for semantic segmentation task.

    Attributes:
        `raw_sample` - get one pair of an image and segmentation mask
        without preprocessing
        `__getitem__` - get one pair of an image and segmentation mask
        for model traning with preprocess by `SegformerImageProcessor` and
        apply transforms.
        `__len__` - get total number of the images in `images_dir`
        (provided in class constructor).
    """

    def __init__(
        self,
        images_dir: Path | str,
        seg_masks_dir: Path | str,
        model_preprocessor: SegformerImageProcessor,
        transforms: Transform | None = None,
    ):
        self._image_dir = Path(images_dir)
        self._target_dir = Path(seg_masks_dir)
        self._img_processor = model_preprocessor
        self.transforms = transforms

        image_files = self._image_dir.glob("Door????.png")
        self._image_files = sorted(image_files)

    def raw_sample(self, idx) -> tuple[Tensor, tv_tensors.Mask]:
        """Get unprocessed pair of image and segmentation mask
        without `SegformerImageProcessor` and applying transforms.

        Args:
            idx: index of sample
        """
        img_path = self._image_files[idx]
        f_name = img_path.name

        # read images into pytorch format directly.
        # not using PIL.Image.open here for better performance
        image = decode_image(img_path, mode=ImageReadMode.RGB)

        # We have only one class for this dataset: "door",
        # so, we can load image as grayscale and make everything more 1 as 1.
        # i.e. assing class 1 to all non-zero pixels
        segmentation_map = decode_image(self._target_dir / f_name, mode=ImageReadMode.GRAY)
        segmentation_map[segmentation_map > 0] = 1
        # Using tv_tensors helps to transform image and mask together
        segmentation_map = tv_tensors.Mask(segmentation_map)

        return image, segmentation_map

    def __getitem__(self, idx):
        """Return preprocessed pair of image and corresponding
        segmentation mask with index `idx`
        """
        image, segmentation_map = self.raw_sample(idx)

        if self.transforms:
            image, segmentation_map = self.transforms(image, segmentation_map)

        encoded_inputs = self._img_processor(
            image, segmentation_map, return_tensors="pt"
        )

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs

    def __len__(self):
        return len(self._image_files)
