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