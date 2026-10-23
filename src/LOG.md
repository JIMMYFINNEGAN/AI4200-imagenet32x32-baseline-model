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
**Result:** Test accuracy decreased from 58.48% to 58.19%. Best validation accuracy increased from 58.68% to 59.06% at epoch 26, but minimum validation loss worsened from 1.1996 to 1.2203. Final training loss decreased to 0.8127, while validation loss rose toward the end of training. Removing dropout improved training fit but did not improve final test performance.

### Change 11 — CIFAR-10 — 2026-10-22

**Change:** Added random horizontal flipping with a 50% probability to the existing random cropping augmentation. Returned dropout to 0.1.
**Why:** Random cropping improved generalization by introducing positional variation. Horizontal flipping may further improve generalization by teaching the model to recognize objects regardless of whether they face left or right.
**Expected:** Higher validation and test accuracy.
**Result:** Test accuracy improved from 58.48% to 58.59%, an increase of 0.11 percentage points. Best validation accuracy increased from 58.68% to 59.36% at epoch 29. Minimum validation loss improved from 1.1996 to 1.1784. Validation loss continued decreasing through the final epochs, suggesting the model may benefit from additional training.

### Change 12 — CIFAR-10 — 2026-10-22

**Change:** Increased training from 30 to 40 epochs and extended the cosine learning-rate schedule from 30 to 40 epochs.
**Why:** With both augmentation techniques, validation loss continued decreasing through the final epochs, suggesting the model may benefit from additional training.
**Expected:** Higher validation and test accuracy, potentially exceeding 59%, without significant overfitting.
**Result:** Test accuracy improved from 58.59% to 59.66%. Best validation accuracy reached 59.96% at epoch 40, compared with 59.36% previously. Minimum validation loss improved slightly to 1.1730 at epoch 28. Additional training improved classification accuracy without severe overfitting.

### Change 13 — CIFAR-10 — 2026-10-22

**Change:** Increased training from 40 to 50 epochs and extended the cosine learning-rate schedule from 40 to 50 epochs.
**Why:** Accuracy increased when epochs were increased to from 30 to 40. Perhaps increasing from 40 to 50 will also yield a higher validation and test accuracy.
**Expected:** Higher validation and test accuracy or potential overfitting.
**Result:** Test accuracy improved from 59.66% to 60.25%. Best validation accuracy reached 60.12% at epoch 43. Validation loss began increasing after epoch 28, indicating some overfitting, but classification accuracy continued improving.

### Change 14 — CIFAR-10 — 2026-10-22

**Change:** Increased the number of hidden neurons in each residual block from 256 to 512.
**Why:** Our model reached 60.25% test accuracy with 256 hidden neurons. Increasing the hidden layer size gives the model more capacity to learn complex patterns and relationships between image pixels.
**Expected:** Higher validation and test accuracy due to increased model capacity, potentially exceeding our current best of 60.25%. However, the larger model may be more prone to overfitting.
**Result:** Test accuracy decreased from 60.25% to 59.64%. Best validation accuracy decreased from 60.12% to 59.74%. Minimum validation loss increased from 1.1772 to 1.1965 at epoch 24. Training loss continued decreasing to 0.6643, while validation loss increased to 1.3194 by epoch 50, indicating greater overfitting. Increasing hidden neurons did not improve generalization.

### Change 15 — MNIST — 2026-10-22

**Change:** Added random cropping with 2 pixels of padding to MNIST training images. Each 28×28 image is padded to 32×32 and randomly cropped back to 28×28.
**Why:** Random cropping introduces small positional variations, helping the model recognize handwritten digits regardless of slight shifts in their position.
**Expected:** Improved generalization and higher validation and test accuracy by making the model less sensitive to digit positioning.
**Result:** Achieved 97.98% test accuracy, compared with the original baseline of 95.06%. Best validation accuracy reached 98.05% at epoch 10, with a minimum validation loss of 0.0619. Validation performance was still improving at the final epoch, suggesting additional training may provide further improvements.

### Change 16 — MNIST — 2026-10-22

**Change:** Increased MNIST training from 10 to 20 epochs and adjusted the cosine learning-rate scheduler to T_max=20.
**Why:** Validation accuracy was still improving at epoch 10, suggesting the model could benefit from additional training.
**Expected:** Higher validation and test accuracy, potentially exceeding the current best of 97.98%. Additional epochs may also introduce overfitting.
**Result:** Test accuracy improved from 97.98% to 98.92%. Best validation accuracy reached 98.92% at epoch 20, and minimum validation loss decreased to 0.0369. Both metrics continued improving through the final epoch, indicating that additional training may provide further gains.

### Change 17 — MNIST — 2026-10-22

**Change:** Increased MNIST training from 20 to 30 epochs and adjusted the cosine learning-rate scheduler to T_max=30. Kept all other settings unchanged.
**Why:** Validation accuracy and validation loss were still improving at epoch 20, suggesting the model could benefit from additional training.
**Expected:** Higher validation and test accuracy, potentially exceeding 99%. Additional epochs may also introduce overfitting.
**Result:** PTest accuracy improved from 98.92% to 99.03%. Best validation accuracy reached 99.12% at epochs 24 and 28, and minimum validation loss reached 0.0317 at epoch 29. Performance improvements began leveling off during the final epochs, suggesting diminishing returns from additional training.