
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from .model import TASKS, ResidualMLP


BATCH_SIZE = 256
EPOCHS = {"mnist": 30, "cifar10": 50, "imagenet32": 20}


def prepare(X):
    """Normalize image pixels to [-1, 1]."""
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


def train(task, X_train, y_train, seed=0, device="cuda"):
    """Train and return a new model."""
    torch.manual_seed(seed)

    # Enable optional ImageNet32 diagnostics.
    run_analysis = (
        task == "imagenet32"
        and os.environ.get("RUN_ANALYSIS") == "1"
    )

    if run_analysis:
        from .analysis import collect_diagnostics, run_ablation

    model = ResidualMLP(**TASKS[task]).to(device)

    # Create training and validation datasets.
    dataset = TensorDataset(
        prepare(X_train),
        torch.as_tensor(y_train).long()
    )

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Set up fixed data for diagnostics.
    if run_analysis:
        fixed_X, fixed_y = next(iter(val_loader))
        fixed_X = fixed_X.to(device)
        fixed_y = fixed_y.to(device)
        diagnostic_history = {}

    # Set up loss, optimizer, and learning rate schedule.
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS[task],
        eta_min=1e-5
    )

    if run_analysis:
        diagnostic_history["initialization"] = collect_diagnostics(
            model, fixed_X, fixed_y, optimizer
        )
        print("Collected initialization diagnostics", flush=True)

    # Train for the specified number of epochs.
    for epoch in range(EPOCHS[task]):
        total_loss = 0.0
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Apply task-specific data augmentation.
            if task == "cifar10":
                images = X_batch.reshape(-1, 3, 32, 32)

                # Pad and randomly crop images.
                images = F.pad(images, (4, 4, 4, 4))

                images = torch.stack([
                    images[i, :, top:top + 32, left:left + 32]
                    for i, (top, left) in enumerate(
                        zip(
                            torch.randint(
                                0, 9, (len(images),)
                            ).tolist(),
                            torch.randint(
                                0, 9, (len(images),)
                            ).tolist()
                        )
                    )
                ])

                # Random horizontal flip.
                flip_mask = (
                    torch.rand(len(images), device=images.device) < 0.5
                )
                images[flip_mask] = images[flip_mask].flip(dims=[3])

                X_batch = images

            elif task == "mnist":
                images = X_batch.reshape(-1, 1, 28, 28)

                # Pad and randomly crop images.
                images = F.pad(
                    images, (2, 2, 2, 2), value=-1.0
                )

                images = torch.stack([
                    images[i, :, top:top + 28, left:left + 28]
                    for i, (top, left) in enumerate(
                        zip(
                            torch.randint(
                                0, 5, (len(images),)
                            ).tolist(),
                            torch.randint(
                                0, 5, (len(images),)
                            ).tolist()
                        )
                    )
                ])

                X_batch = images

            elif task == "imagenet32":
                images = X_batch.reshape(-1, 3, 32, 32)

                # Reflection padding and random cropping.
                images = F.pad(
                    images, (4, 4, 4, 4), mode="reflect"
                )

                images = torch.stack([
                    images[i, :, top:top + 32, left:left + 32]
                    for i, (top, left) in enumerate(
                        zip(
                            torch.randint(
                                0, 9, (len(images),)
                            ).tolist(),
                            torch.randint(
                                0, 9, (len(images),)
                            ).tolist()
                        )
                    )
                ])

                # Random horizontal flip.
                flip_mask = (
                    torch.rand(len(images), device=images.device) < 0.5
                )
                images[flip_mask] = images[flip_mask].flip(dims=[3])

                X_batch = images

            # Forward pass and weight update.
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        # Evaluate on the validation set.
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)

                val_loss += loss.item() * len(X_batch)
                val_correct += (
                    (logits.argmax(dim=1) == y_batch).sum().item()
                )

        # Print training and validation results.
        print(
            f"epoch {epoch + 1}: "
            f"train loss {total_loss / len(train_dataset):.4f}, "
            f"val loss {val_loss / len(val_dataset):.4f}, "
            f"val accuracy {val_correct / len(val_dataset):.2%}",
            flush=True,
        )

        scheduler.step()

        # Collect diagnostics at selected epochs.
        if run_analysis and (epoch + 1) in (2, 10, 20):
            stage = f"epoch_{epoch + 1}"

            diagnostic_history[stage] = collect_diagnostics(
                model, fixed_X, fixed_y, optimizer
            )

            print(f"Collected diagnostics: {stage}", flush=True)

            # Save progress in case training stops.
            with open("imagenet32_diagnostics.json", "w") as f:
                json.dump(diagnostic_history, f, indent=2)

    if run_analysis:
        # Save final diagnostic results.
        with open("imagenet32_diagnostics.json", "w") as f:
            json.dump(diagnostic_history, f, indent=2)

        print("Saved imagenet32_diagnostics.json", flush=True)

        # Measure the effect of removing each residual block.
        print(
            "Starting 30-block validation ablation; "
            "this can take additional time.",
            flush=True
        )

        ablation_results = run_ablation(
            model, val_loader, device
        )

        with open("imagenet32_ablation.json", "w") as f:
            json.dump(ablation_results, f, indent=2)

        print("Saved imagenet32_ablation.json", flush=True)

    return model


def predict(model, X):
    """Return class probabilities for each input image."""
    device = next(model.parameters()).device
    X = prepare(X)

    # Predict in batches to limit memory usage.
    chunks = [
        model.predict(X[i:i + 1024].to(device)).cpu()
        for i in range(0, len(X), 1024)
    ]

    return torch.cat(chunks)


