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

### Change 2 — validation testing — 2026-10-21

**Change:** Split training set into 90% training, 10% validation.
**Why:** Tracking validation loss and accuracy after each epoch allows us to monitor model performance and look for overfitting.
**Expected:** More insight into how model is doing on unseen data.
**Result:** Best validation accuracy 53.00%, lowest validation loss 1.3916 at epoch 8, final test accuracy 52.38%, and evidence of overfitting after roughly epoch 8.

### Change 3 — GELU — 2026-10-21

**Change:** Replaced tanh with GELU activation
**Why:** GELU may improve model by reducing saturation and preserving more information through the network.
**Expected:** GELU may improve validation accuracy by reducing saturation and allowing better gradient flow. However, it may increase overfitting.
**Result:** CIFAR-10 test accuracy improved from    52.38% to 55.03%. Best validation accuracy was    55.16% at epoch 9, and lowest validation loss was  1.3087 at epoch 7. GELU reduced training loss faster than tanh but showed substantial overfitting after approximately epoch 7.

### Change 4 — dropout — 2026-10-21

**Change:** Added dropout with probability 0.2 after GELU in each residual block.
**Why:** GELU showed substantial overfitting with validation loss increasing after epoch 7. Dropout will make it harder for the model to memorize the training images.
**Expected:** Reduced overfitting and improved validation accuracy.
**Result:** Test accuracy fell from 55.03% to 52.82% but the best validation loss increased from 55.16% to 56.48% and achieved best validaiton loss at epoch 12. Training loss went up and validation loss went down showing overfitting was improved.

### Change 4 — epochs — 2026-10-21

**Change:** Changing cifar-10 epochs to 12.
**Why:** We found our best validation loss at epoch 12 before the model began to overfit.
**Expected:** Possibly better test accuracy.
**Result:** T