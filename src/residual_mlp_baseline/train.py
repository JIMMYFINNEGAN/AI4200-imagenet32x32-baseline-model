import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from .model import TASKS, ResidualMLP


# Baseline settings. Deliberately plain so they are easy to beat:
# plain SGD, constant learning rate, no dropout, no schedule, no augmentation.
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = {"mnist": 10, "cifar10": 50, "imagenet32": 10}


def prepare(X):
    """Convert uint8 images (0-255) to floats in roughly [-1, 1]."""
    return (torch.as_tensor(X).float() / 255.0 - 0.5) / 0.5


# ---------------------------------------------------------------------------
# The two functions below are the REQUIRED INTERFACE for every submission.
# The grading notebook calls them like this:
#
#     model = train("cifar10", X_train, y_train, seed=0, device="cuda")
#     probs = predict(model, X_test)
#
# Change anything inside them, but keep their names and arguments.
# ---------------------------------------------------------------------------

def train(task, X_train, y_train, seed=0, device="cuda"):
    """Build a new model and train it from scratch. Returns the trained model."""
    torch.manual_seed(seed)                   # makes the run reproducible

    # 1. Build the model for this task. Only input size and class count differ.
    model = ResidualMLP(**TASKS[task]).to(device)

    # 2. Wrap the data so we can loop over shuffled mini-batches.
    # 2. Split the provided training data into training and validation sets.
    dataset = TensorDataset(prepare(X_train), torch.as_tensor(y_train).long())

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
         generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Choose the loss and the optimizer.
    loss_fn = nn.CrossEntropyLoss()           # expects raw logits
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min = 1e-5)
    # 4. The training loop.
    for epoch in range(EPOCHS[task]):
        total_loss = 0.0

        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if task == "cifar10":
                images = X_batch.reshape(-1, 3, 32, 32)

                 # Add 4 pixels of padding around each image.
                images = F.pad(images, (4, 4, 4, 4))

                 # Randomly crop each image back to 32x32.
                images = torch.stack([
                    images[i, :, top:top + 32, left:left + 32]
                    for i, (top, left) in enumerate(
                        zip(
                            torch.randint(0, 9, (len(images),)).tolist(),
                            torch.randint(0, 9, (len(images),)).tolist()
                        )
                    )
                ])

                flip_mask = torch.rand(len(images), device=images.device) < 0.5
                images[flip_mask] = images[flip_mask].flip(dims=[3])
                
                X_batch = images

            logits = model(X_batch)           # forward pass
            loss = loss_fn(logits, y_batch)   # how wrong are we?

            optimizer.zero_grad()             # clear old gradients
            loss.backward()                   # backward pass: compute gradients
            optimizer.step()                  # update the weights

            total_loss += loss.item() * len(X_batch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)

                val_loss += loss.item() * len(X_batch)
                val_correct += (logits.argmax(dim=1) == y_batch).sum().item()

        print(
            f"epoch {epoch + 1}: "
            f"train loss {total_loss / len(train_dataset):.4f}, "
            f"val loss {val_loss / len(val_dataset):.4f}, "
            f"val accuracy {val_correct / len(val_dataset):.2%}"
        )
        scheduler.step()

    return model


def predict(model, X):
    """Return class probabilities for X, one row per example, on the CPU."""
    device = next(model.parameters()).device
    X = prepare(X)
    # Predict in chunks so a large test set does not run out of GPU memory.
    chunks = [model.predict(X[i:i + 1024].to(device)).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(chunks)
