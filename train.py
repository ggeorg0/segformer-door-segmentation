import warnings     # mute `evaluate`'s Mean IoU metric warnings
from pathlib import Path
from time import time

import evaluate
import torch
from torch import Generator
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from tqdm.notebook import tqdm
# from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from data_utils import DoorsDataset

# defining constants
DATASET_PATH = Path("/home/ggeorge/datasets/DeepDoors_2/Detection_Segmentation")

IMGS_DIR = "Images"
SEG_MAPS_DIR = "Annotations"

N_EPOCHS = 5

torch.manual_seed(127)

# segformer image preprocessor: adjust resolution, normalize and so on.
img_processor = SegformerImageProcessor()

dataset = DoorsDataset(
    images_dir=DATASET_PATH / IMGS_DIR,
    seg_masks_dir=DATASET_PATH / SEG_MAPS_DIR,
    model_preprocessor=img_processor,
    # transforms=transforms
)

id2label = {0: "background", 1: "door"}

# make value as key and key as value
label2id = {v: k for k, v in id2label.items()}

# define model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [0.80, 0.10, 0.10], generator=Generator().manual_seed(127)
)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=4)
test_dataloder = DataLoader(test_dataset, batch_size=4, num_workers=4)

n_epochs = 1

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)

# Mean IoU metric
mean_iou = evaluate.load("mean_iou")

# move model to GPU
device = torch.device("cuda")
model.to(device)

print(f"Starting traning, using {device}")

for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    print("Epoch:", epoch)

    model.train()

    train_loss = 0.0
    train_t0 = time()
    for idx, batch in enumerate(tqdm(train_dataloader)):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / len(train_dataloader)
    print(f"\tTraining loss: {avg_train_loss:.6f}")
    train_t1 = time()

    model.eval()

    val_loss = 0.0
    val_t0 = time()
    with torch.no_grad():
        for batch in val_dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            val_loss += outputs.loss.item()

            logits = outputs.logits

            # Upsample logits to match label size (SegFormer outputs lower resolution)
            upsampled_logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            # Shape: (batch_size, num_labels, H, W)
            predicted = upsampled_logits.argmax(dim=1)

            mean_iou.add_batch(
                predictions=predicted.detach().cpu().numpy().astype("int32"),
                references=labels.detach().cpu().numpy().astype("int32"),
            )

    avg_val_loss = val_loss / len(val_dataloader)

    # Sometimes 'RuntimeWarning: invalid value encountered in divide' occurs here
    # I haven't fully figured out why, but I assume this is due to the fact
    # sometimes there are no door pixels at all.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        val_metrics = mean_iou.compute(
            num_labels=len(id2label), ignore_index=0, reduce_labels=False
        )

    # Print results
    print(f"\tValidation Loss: {avg_val_loss:.6f}")
    print(f"\tValidation Mean IoU: {val_metrics['mean_iou']:.6f}")
    print(f"\tValidation Mean Accuracy: {val_metrics['mean_accuracy']:.6f}")
    print(f"\n\tEpoch train time: {train_t1 - train_t0:.6f} sec.")
    print(f"\tEpoch validation time: {time() - val_t0:.6f} sec.")
    print("---" * 15)
