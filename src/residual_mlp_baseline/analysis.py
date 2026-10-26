import torch


def rms(x):
    """Root mean square of a tensor."""
    return x.detach().float().square().mean().sqrt().item()


def measure_activations(model, X_batch):
    """
    Measure activation variance and residual branch-to-stream
    ratio at each residual block.
    """
    model.eval()

    variances = []
    branch_ratios = []

    def make_hook():
        def hook(module, inputs, output):
            x = inputs[0]
            residual = output - x

            variances.append(
                output.detach().float().var(dim=0, unbiased=False).mean().item()
            )

            branch_ratios.append(
                rms(residual) / max(rms(x), 1e-12)
            )

        return hook

    handles = [
        block.register_forward_hook(make_hook())
        for block in model.blocks
    ]

    try:
        with torch.no_grad():
            model(X_batch)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "activation_variance": variances,
        "branch_to_stream": branch_ratios,
    }


def measure_gradients(model):
    """
    Measure the RMS gradient of each residual block's weights.
    Call this after loss.backward() and before optimizer.zero_grad().
    """
    gradient_rms = []

    for block in model.blocks:
        gradients = []

        for parameter in block.parameters():
            if parameter.grad is not None and parameter.ndim >= 2:
                gradients.append(parameter.grad.detach().flatten().float())

        if gradients:
            combined = torch.cat(gradients)
            gradient_rms.append(rms(combined))
        else:
            gradient_rms.append(float("nan"))

    return gradient_rms


def snapshot_weights(model):
    """
    Save copies of each residual block's weight matrices
    immediately before an optimizer step.
    """
    snapshots = []

    for block in model.blocks:
        weights = [
            parameter.detach().clone()
            for parameter in block.parameters()
            if parameter.ndim >= 2
        ]
        snapshots.append(weights)

    return snapshots


def measure_updates(model, snapshots):
    """
    Calculate RMS(weight change) / RMS(weight before update).
    Call immediately after optimizer.step().
    """
    update_ratios = []

    for block, old_weights in zip(model.blocks, snapshots):
        new_weights = [
            parameter.detach()
            for parameter in block.parameters()
            if parameter.ndim >= 2
        ]

        old = torch.cat([w.flatten().float() for w in old_weights])
        new = torch.cat([w.flatten().float() for w in new_weights])

        delta = new - old
        ratio = rms(delta) / max(rms(old), 1e-12)

        update_ratios.append(ratio)

    return update_ratios
