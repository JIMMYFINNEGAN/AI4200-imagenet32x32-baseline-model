### Change 0 — baseline — 2026-10-20

**Result:** 
MNIST: 95.06% test accuracy
CIFAR-10: 48.61% test accuracy
ImageNet 32x32: 6.77% test accuracy

### Change 1 — AdamW optimizer — 2026-10-21

**Change:** Replaced SGD (lr 0.05) with AdamW (lr 1e-3, weight decay 0.01).
**Why:** AdamW uses adaptive learning rates and momentum estimates, which may improve optimization compared to plain SGD.
**Expected:** AdamW should converge faster and improve CIFAR-10 validation accuracy compared to the baseline.
**Result:** Test accuracy 48.61% -> 52.4%. 1.7 minute train time. The AdamW training configuration improved performance.

### Change 12 — cifar10 — 2026-10-20

**Change:** Replaced SGD (lr 0.05) with AdamW (lr 1e-3, weight decay 0.05).
**Why:** Train loss was still falling slowly after 30 epochs, so I think the
optimizer, not model capacity, is the bottleneck.
**Expected:** Faster convergence and higher test accuracy, maybe +3 to 5 points.
**Result:** Test accuracy 48.1% -> 53.4%. Train accuracy 71%. 14 min on one GPU.
The gain was as expected, but the train/test gap grew from 9 to 18 points,
so the model is now overfitting. Next I will add dropout.