
import copy
import torch
import torch.nn.functional as F


def rms(x):
    """Root mean square of a tensor."""
    return x.detach().float().square().mean().sqrt().item()


def measure_activations(model, X_batch):
    """Per-block residual-stream variance and residual-to-stream RMS ratio."""
    was_training = model.training
    variances = []
    branch_ratios = []

    def hook(module, inputs, output):
        stream = inputs[0]
        residual = output - stream
        variances.append(
            output.detach().float().var(dim=0, unbiased=False).mean().item()
        )
        branch_ratios.append(rms(residual) / max(rms(stream), 1e-12))

    handles = [block.register_forward_hook(hook) for block in model.blocks]
    try:
        model.eval()
        with torch.no_grad():
            model(X_batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return {
        "activation_variance": variances,
        "branch_to_stream": branch_ratios,
    }


def measure_gradients(model):
    """RMS of all weight-matrix gradients in each residual block."""
    result = []
    for block in model.blocks:
        grads = [
            p.grad.detach().flatten().float()
            for p in block.parameters()
            if p.ndim >= 2 and p.grad is not None
        ]
        result.append(rms(torch.cat(grads)) if grads else float("nan"))
    return result


def snapshot_weights(model):
    """Copy each block's weight matrices before an optimizer update."""
    return [
        [p.detach().clone() for p in block.parameters() if p.ndim >= 2]
        for block in model.blocks
    ]


def measure_updates(model, snapshots):
    """Per-block RMS(delta weights) / RMS(previous weights)."""
    ratios = []
    for block, previous in zip(model.blocks, snapshots):
        current = [
            p.detach() for p in block.parameters() if p.ndim >= 2
        ]
        before = torch.cat([p.flatten().float() for p in previous])
        after = torch.cat([p.flatten().float() for p in current])
        ratios.append(rms(after - before) / max(rms(before), 1e-12))
    return ratios


def collect_diagnostics(model, X_batch, y_batch, optimizer):
    """
    Evaluate a fixed batch and simulate one reversible AdamW step.

    Restores parameters, optimizer state, module mode, and CPU/CUDA RNG state.
    The update ratio describes a hypothetical diagnostic step, not an actual
    training minibatch update.
    """
    was_training = model.training
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None
    )

    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())

    try:
        activations = measure_activations(model, X_batch)

        model.eval()
        optimizer.zero_grad(set_to_none=True)

        loss = F.cross_entropy(model(X_batch), y_batch)
        loss.backward()

        gradients = measure_gradients(model)

        before = snapshot_weights(model)
        optimizer.step()
        updates = measure_updates(model, before)

        return {
            **activations,
            "gradient_rms": gradients,
            "update_to_weight": updates,
        }

    finally:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        optimizer.zero_grad(set_to_none=True)
        model.train(was_training)

        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def validation_accuracy(model, val_loader, device):
    """Compute accuracy over the full held-out validation split."""
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            predictions = model(X).argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.numel()

    return correct / total


def run_ablation(model, val_loader, device):
    """
    Skip one residual block at a time and measure validation accuracy drop.
    """
    was_training = model.training
    model.eval()

    try:
        baseline = validation_accuracy(model, val_loader, device)
        results = []

        for index, block in enumerate(model.blocks):
            # Return the skip connection input instead of x + branch(x).
            handle = block.register_forward_hook(
                lambda module, inputs, output: inputs[0]
            )

            try:
                accuracy = validation_accuracy(model, val_loader, device)
            finally:
                handle.remove()

            result = {
                "block": index + 1,
                "accuracy": accuracy,
                "accuracy_drop_percentage_points": 100.0 * (
                    baseline - accuracy
                ),
            }

            results.append(result)

            print(
                f"Ablation block {index + 1:02d}/{len(model.blocks)}: "
                f"val accuracy {accuracy:.2%}, "
                f"drop {result['accuracy_drop_percentage_points']:.3f} pp",
                flush=True,
            )

        return {
            "baseline_accuracy": baseline,
            "blocks": results,
        }

    finally:
        model.train(was_training)
