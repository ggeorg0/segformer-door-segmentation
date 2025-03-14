import warnings
from pathlib import Path
from time import time

import evaluate
import mlflow
import numpy as np
import torch
from torch import Generator
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from tqdm import tqdm

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from data_utils import DoorsDataset

# defining constants
DATASET_PATH = Path("/home/ggeorge/datasets/DeepDoors_2/Detection_Segmentation")
IMGS_DIR = "Images"
SEG_MAPS_DIR = "Annotations"
N_EPOCHS = 50
LR = 0.00006
BATCH_SIZE = 8
MLFLOW_EXP_ID = "639229507917248954"

mlflow.set_tracking_uri("http://localhost:17888")

torch.manual_seed(127)

# segformer image preprocessor: set right resolution, normalize values and so on.
img_processor = SegformerImageProcessor()
color_jitter = v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
scale_jitter = v2.ScaleJitter(target_size=(512, 512), scale_range=(0.1, 2))
transforms = v2.Compose([
    color_jitter,
    scale_jitter,
    v2.RandomHorizontalFlip(p=0.5)
])

dataset = DoorsDataset(
    images_dir=DATASET_PATH / IMGS_DIR,
    seg_masks_dir=DATASET_PATH / SEG_MAPS_DIR,
    model_preprocessor=img_processor,
    transforms=transforms
)

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [0.80, 0.10, 0.10], generator=Generator().manual_seed(127)
)

id2label = {0: "background", 1: "door"}
label2id = {v: k for k, v in id2label.items()}


# define model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
device = torch.device("cuda")
model.to(device)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=4)

n_epochs = 1

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Mean IoU metric
mean_iou = evaluate.load("mean_iou")

def train_model(
    model: SegformerForSemanticSegmentation,
    device: torch.device,
    data_loader: DataLoader,
) -> float:
    """Train model one epoch using given `data_loader`
    and return mean loss"""
    model.train()

    train_loss = 0.0
    for batch in tqdm(data_loader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    mean_loss = train_loss / len(train_dataloader)
    return mean_loss

def eval_loss_n_metrics(
    model: SegformerForSemanticSegmentation,
    device: torch.device,
    data_loader: DataLoader,
) -> tuple[float, dict]:
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss += outputs.loss.item()

            logits = outputs.logits

            # Upsample logits to match label size (SegFormer outputs lower resolution)
            upsampled_logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)

            mean_iou.add_batch(
                # converting to uint8 prevents warnings
                predictions=predicted.detach().cpu().numpy().astype(np.uint8),
                references=labels.detach().cpu().numpy().astype(np.uint8),
            )

    # Sometimes `RuntimeWarning: invalid value encountered in divide` occurs here
    # I haven't fully figured out why, but I assume this is due to the fact
    # sometimes there are no door pixels at all.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        metrics = mean_iou.compute(
            num_labels=len(id2label), ignore_index=0, reduce_labels=False
        )

    if metrics is None:
        raise ValueError("IOU metrics is not calculated!")

    mean_loss = loss / len(data_loader)
    return mean_loss, metrics

with mlflow.start_run(experiment_id=MLFLOW_EXP_ID):
    # Log hyperparameters
    mlflow.log_param("N_EPOCHS", N_EPOCHS)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("batch_size", BATCH_SIZE)

    for epoch in range(N_EPOCHS):
        print("Epoch:", epoch)

        # Training phase
        t0 = time()
        train_loss = train_model(model, device, train_dataloader)
        train_time = time() - t0

        # Validation phase
        t0 = time()
        val_loss, val_metrics = eval_loss_n_metrics(model, device, val_dataloader)
        val_time = time() - t0

        # Log metrics with epoch step
        mlflow.log_metrics({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/mean_iou": val_metrics['mean_iou'],
            "val/mean_accuracy": val_metrics['mean_accuracy'],
            "time/train": train_time,
            "time/val": val_time
        }, step=epoch)

        print(f"\tTraining loss: {train_loss:.6f}")
        print(f"\tValidation Loss: {val_loss:.6f}")
        print(f"\tValidation Mean IoU: {val_metrics['mean_iou']:.6f}")
        print(f"\tValidation Mean Accuracy: {val_metrics['mean_accuracy']:.6f}")
        print(f"\n\tEpoch train time: {train_time:.6f} sec.")
        print(f"\tEpoch validation time: {val_time:.6f} sec.")
        print("---" * 15)

        # Save model checkpoint and log as artifact
        if epoch % 5 == 0:
            model_save_dir = f'./tuned-segformer-epoch-{epoch}'
            model.save_pretrained(model_save_dir)
            mlflow.log_artifacts(model_save_dir, artifact_path=f"model_checkpoints/epoch_{epoch}")

    # Final test evaluation
    loss, test_metrics = eval_loss_n_metrics(model, device, test_dataloader)
    mlflow.log_metrics({
        "test/loss": loss,
        "test/mean_iou": test_metrics['mean_iou'],
        "test/mean_accuracy": test_metrics['mean_accuracy']
    })

    print(f"Test Loss: {loss:.6f}")
    print(f"Test Mean IoU: {test_metrics['mean_iou']:.6f}")
    print(f"Test Mean Accuracy: {test_metrics['mean_accuracy']:.6f}")
