### Change 0 — baseline — 2026-10-20

**Result:** 
MNIST: 95.06% test accuracy
CIFAR-10: 48.61% test accuracy
ImageNet 32x32: 6.77% test accuracy

### Change 1 — cifar — 2026-10-21

**Change:** Replaced SGD (lr 0.05) with AdamW (lr 1e-3, weight decay 0.01).
**Why:** AdamW uses adaptive learning rates and momentum estimates, which may improve optimization compared to plain SGD.
**Expected:** AdamW should converge faster and improve CIFAR-10 validation accuracy compared to the baseline.
**Result:** Test accuracy 48.61% -> 52.4%. 1.7 minute train time. The AdamW training configuration improved performance.

### Change 2 — cifar — 2026-10-21

**Change:** Split training set into 90% training, 10% validation.
**Why:** Tracking validation loss and accuracy after each epoch allows us to monitor model performance and look for overfitting.
**Expected:** More insight into how model is doing on unseen data.
**Result:** Best validation accuracy 53.00%, lowest validation loss 1.3916 at epoch 8, final test accuracy 52.38%, and evidence of overfitting after roughly epoch 8.

### Change 3 — cifar — 2026-10-21

**Change:** Replaced tanh with GELU activation
**Why:** GELU may improve model by reducing saturation and preserving more information through the network.
**Expected:** GELU may improve validation accuracy by reducing saturation and allowing better gradient flow. However, it may increase overfitting.
**Result:** CIFAR-10 test accuracy improved from    52.38% to 55.03%. Best validation accuracy was    55.16% at epoch 9, and lowest validation loss was  1.3087 at epoch 7. GELU reduced training loss faster than tanh but showed substantial overfitting after approximately epoch 7.

### Change 4 — cifar — 2026-10-21

**Change:** Added dropout with probability 0.2 after GELU in each residual block.
**Why:** GELU showed substantial overfitting with validation loss increasing after epoch 7. Dropout will make it harder for the model to memorize the training images.
**Expected:** Reduced overfitting and improved validation accuracy.
**Result:** Test accuracy fell from 55.03% to 52.82% but the best validation loss increased from 55.16% to 56.48% and achieved best validaiton loss at epoch 12. Training loss went up and validation loss went down showing overfitting was improved.

### Change 5 — cifar — 2026-10-21

**Change:** Changing cifar-10 epochs from 30 to 12.
**Why:** We found our best validation loss at epoch 12 before the model began to overfit.
**Expected:** Possibly better test accuracy.
**Result:** Test accuracy increased from 52.82% to 55.40%.

### Change 6 — cifar — 2026-10-21

**Change:** Replaced GELU with SwiGLU in the residual blocks and kept GELU in classification head.
**Why:** SwiGLU has a learned gating mechanism and may improve network's ability to represent useful features.
**Expected:** Potentially improve validation accuracy but may increase overfitting.
**Result:** Test accuracy was 54.62% after 30 epochs. Best validation accuracy was 54.60% and lowest validation loss was 1.3793 at epoch 5. Performance was below the GELU + dropout configuration, with evidence of substantial overfitting.

### Change 7 — cifar — 2026-10-22

**Change:** Returning to GELU with dropout(0.2) and adding a cosine learning rate scheduler over 30 epochs.
**Why:** GELU was performing well in the early epochs before overfitting. A learning rate scheduler reduces the learning rate and allows smaller parameter updates.
**Expected:** The validation accuracy will improve over a larger number of epochs.
**Result:** Achieved 56.17% test accuracy, improving upon the previous best of 55.40%. Best validation accuracy reached 56.84% at epoch 19. Validation loss still increased after epoch 8, indicating continued overfitting.

### Change 8 — CIFAR-10 — 2026-10-22

**Change:** Added random cropping data augmentation to CIFAR-10 training images. Each image receives 4 pixels of padding on every side and is randomly cropped back to 32×32.
**Why:** Our MLP learns from individual pixel positions and may struggle to recognize objects when they appear in slightly different locations. Random cropping introduces positional variation during training to improve generalization.
**Expected:** Reduce overfitting and improve validation and test accuracy beyond our current best of 56.17%.
**Result:** Test accuracy improved from 56.17% to 57.94%. The best validation accuracy was 58.30% at epoch 28. Validation loss continued decreasing throughout most of training, reaching a minimum of 1.1970. This suggests substantially better generalization and less overfitting.

### Change 9 — CIFAR-10 — 2026-10-22

**Change:** Lowering dropout to a probability of 0.1 in residual blocks and head.
**Why:** We added dropout to prevent overfitting. Now that we've used data augmentation, our model is still learning in later epochs. We may be regularizing more than necessary.
**Expected:** Higher validation accuracy and potentially test accuracy.
**Result:** Test accuracy increased from 57.94% to 58.48%. Best validation accuracy improved from 58.30% to 58.68%. Reducing dropout improved classification accuracy without causing severe overfitting.

### Change 10 — CIFAR-10 — 2026-10-22

**Change:** Reduced dropout probability from 0.1 to 0.0 in the residual blocks and classification head.
**Why:** Reducing dropout from 0.2 to 0.1 improved test accuracy from 57.94% to 58.48%. Random cropping has substantially reduced overfitting, so dropout may no longer be necessary.
**Expected:** Potentially higher validation and test accuracy as the model learns without dropout restrictions. However, validation loss may increase if overfitting returns.
**Result:** Pending.