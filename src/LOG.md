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

### Change 8 — cifar — 2026-10-22

**Change:** Added random cropping data augmentation to CIFAR-10 training images. Each image receives 4 pixels of padding on every side and is randomly cropped back to 32×32.
**Why:** Our MLP learns from individual pixel positions and may struggle to recognize objects when they appear in slightly different locations. Random cropping introduces positional variation during training to improve generalization.
**Expected:** Reduce overfitting and improve validation and test accuracy beyond our current best of 56.17%.
**Result:** Test accuracy improved from 56.17% to 57.94%. The best validation accuracy was 58.30% at epoch 28. Validation loss continued decreasing throughout most of training, reaching a minimum of 1.1970. This suggests substantially better generalization and less overfitting.

### Change 9 — cifar — 2026-10-22

**Change:** Lowering dropout to a probability of 0.1 in residual blocks and head.
**Why:** We added dropout to prevent overfitting. Now that we've used data augmentation, our model is still learning in later epochs. We may be regularizing more than necessary.
**Expected:** Higher validation accuracy and potentially test accuracy.
**Result:** Test accuracy increased from 57.94% to 58.48%. Best validation accuracy improved from 58.30% to 58.68%. Reducing dropout improved classification accuracy without causing severe overfitting.

### Change 10 — cifar — 2026-10-22

**Change:** Reduced dropout probability from 0.1 to 0.0 in the residual blocks and classification head.
**Why:** Reducing dropout from 0.2 to 0.1 improved test accuracy from 57.94% to 58.48%. Random cropping has substantially reduced overfitting, so dropout may no longer be necessary.
**Expected:** Potentially higher validation and test accuracy as the model learns without dropout restrictions. However, validation loss may increase if overfitting returns.
**Result:** Test accuracy decreased from 58.48% to 58.19%. Best validation accuracy increased from 58.68% to 59.06% at epoch 26, but minimum validation loss worsened from 1.1996 to 1.2203. Final training loss decreased to 0.8127, while validation loss rose toward the end of training. Removing dropout improved training fit but did not improve final test performance.

### Change 11 — cifar — 2026-10-22

**Change:** Added random horizontal flipping with a 50% probability to the existing random cropping augmentation. Returned dropout to 0.1.
**Why:** Random cropping improved generalization by introducing positional variation. Horizontal flipping may further improve generalization by teaching the model to recognize objects regardless of whether they face left or right.
**Expected:** Higher validation and test accuracy.
**Result:** Test accuracy improved from 58.48% to 58.59%, an increase of 0.11 percentage points. Best validation accuracy increased from 58.68% to 59.36% at epoch 29. Minimum validation loss improved from 1.1996 to 1.1784. Validation loss continued decreasing through the final epochs, suggesting the model may benefit from additional training.

### Change 12 — cifar — 2026-10-22

**Change:** Increased training from 30 to 40 epochs and extended the cosine learning-rate schedule from 30 to 40 epochs.
**Why:** With both augmentation techniques, validation loss continued decreasing through the final epochs, suggesting the model may benefit from additional training.
**Expected:** Higher validation and test accuracy, potentially exceeding 59%, without significant overfitting.
**Result:** Test accuracy improved from 58.59% to 59.66%. Best validation accuracy reached 59.96% at epoch 40, compared with 59.36% previously. Minimum validation loss improved slightly to 1.1730 at epoch 28. Additional training improved classification accuracy without severe overfitting.

### Change 13 — cifar — 2026-10-22

**Change:** Increased training from 40 to 50 epochs and extended the cosine learning-rate schedule from 40 to 50 epochs.
**Why:** Accuracy increased when epochs were increased to from 30 to 40. Perhaps increasing from 40 to 50 will also yield a higher validation and test accuracy.
**Expected:** Higher validation and test accuracy or potential overfitting.
**Result:** Test accuracy improved from 59.66% to 60.25%. Best validation accuracy reached 60.12% at epoch 43. Validation loss began increasing after epoch 28, indicating some overfitting, but classification accuracy continued improving.

### Change 14 — cifar — 2026-10-22

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

### Change 18 — imagenet — 2026-10-23

**Change:** Using our best performing cifar and mnist model on imagenet. Using GELU activation, dropout of 0.1, AdamW optimizer, cosine lr scheduler.
**Why:** To establish a new baseline on imagenet before making changes.
**Expected:** Higher test accuracy than original MLP baseline model provided.
**Result:** Achieved 9.84% test accuracy. Best validation accuracy reached 10.91% at epoch 10, with minimum validation loss of 5.1308 at epoch 8. Validation accuracy continued improving through the final epoch.

### Change 19 — imagenet — 2026-10-23

**Change:** Replaced GELU activation with SwiGLU in all 30 residual blocks while keeping GELU in the output head.
**Why:** We want to determine whether SwiGLU improves performance under the same training conditions.
**Expected:** Higher validation and test accuracy than the current GELU model, potentially exceeding our previous best of 10%.
**Result:** Test accuracy improved from 9.84% to 10.07%. Best validation accuracy increased from 10.91% to 11.21% at epoch 8, while minimum validation loss decreased from 5.1308 to 5.1082 at epoch 6. Training loss decreased to 4.2555, but validation loss increased to 5.2176 by epoch 10, indicating greater overfitting. SwiGLU improved accuracy slightly but increased training time.

### Change 20 — imagenet — 2026-10-24

**Change:** Added data augmentation to ImageNet32 training using random cropping with 4-pixel reflection padding and horizontal flipping with a 50% probability.
**Why:** The previous SwiGLU experiment showed signs of overfitting, with validation loss increasing after epoch 6. Data augmentation introduces variations in image positioning and orientation, potentially improving generalization.
**Expected:** Improved validation and test accuracy compared with the previous SwiGLU result of 10.07%, along with reduced overfitting.
**Result:** Test accuracy improved from 10.07% to 10.82%. Best validation accuracy reached 12.11%, with a final validation loss of 4.9775. Training took 20.1 minutes. Validation loss continued decreasing throughout the 10 epochs, suggesting improved generalization.

### Change 21 — imagenet — 2026-10-24

**Change:** Removed the cosine learning-rate scheduler.
**Why:** Wanted to determine whether the learning-rate scheduler was contributing to improved performance.
**Expected:** Similar or possibly improved accuracy if maintaining a constant learning rate allowed the model to continue learning effectively.
**Result:** Test accuracy decreased slightly from 10.82% to 10.72%. Best validation accuracy decreased from 12.11% to 11.85%, with a final validation loss of 4.9878. Training took approximately 20 minutes.

### Change 22 — imagenet — 2026-10-24

**Change:** Increased the hidden dimension from 256 to 512.
**Why:** A larger hidden dimension could increase the model's capacity to learn complex features across ImageNet32's 1,000 classes.
**Expected:** Improved accuracy from increased representational capacity, potentially at the cost of additional computation or overfitting.
**Result:** Test accuracy increased from 10.82% to 11.76%. Best validation accuracy increased from 12.11% to 13.00%, with a final validation loss of 4.9160. Training took 20.2 minutes.

### Change 23 — imagenet — 2026-10-25

**Change:** Increased training duration from 10 to 15 epochs.
**Why:** The previous experiment showed continued improvement at the end of training, suggesting that the model had not fully converged.
**Expected:** Higher validation and test accuracy from additional optimization.
**Result:** Test accuracy improved from 11.76% to 12.95%. Best validation accuracy increased from 13.00% to 14.29%, with a final validation loss of 4.8229. Training took 29.9 minutes.

### Change 24 — imagenet — 2026-10-25

**Change:** Increased training duration from 15 to 20 epochs, keeping all other settings unchanged.
**Why:** Validation accuracy was still improving after 15 epochs, indicating that additional training could be beneficial.
**Expected:** Further accuracy improvements, although validation loss might eventually plateau as overfitting increased.
**Result:** Test accuracy improved from 12.95% to 13.29%. Best validation accuracy reached 14.93%, and the lowest validation loss was 4.7963 at epoch 17. Training took 40.2 minutes. Validation loss plateaued toward the end, while validation accuracy continued improving slightly.

### Change 25 — imagenet — 2026-10-26

**Change:** Increased model width from 128 to 256.
**Why:** Wanted to determine whether a wider residual stream could improve feature representation and classification accuracy.
**Expected:** Improved validation accuracy from increased model capacity, with a possible increase in training time.
**Result:** The experiment was interrupted by the one-hour HPC allocation limit before completing 20 epochs. At epoch 13, validation accuracy reached 14.63% with validation loss 4.7678, compared with 14.13% accuracy and 4.8275 loss for the width-128 model at the same epoch. The wider model showed promising intermediate results but required substantially more computation.

### Change 26 — cifar — 2026-10-26

**Change:** Introduced 8×8 image patches with learned patch embeddings and spatial mixing. Each 32×32 RGB image was divided into 16 patches, processed with linear layers, and combined into a width-128 representation before passing through the existing 30 residual blocks.
**Why:** The original flattened MLP did not explicitly preserve local image regions. Patch processing could help the network learn local visual features and relationships between image regions.
**Expected:** Improved classification accuracy through a more structured image representation, without using prohibited convolutional layers.
**Result:** Test accuracy reached 62.15%, exceeding the previous CIFAR-10 best of 60.25%. Best validation accuracy was 62.62% at epoch 34. Minimum validation loss was 1.1419 at epoch 20, while final validation loss increased to 1.7891, indicating substantial overfitting. Training took 4.9 minutes. Because the previous best used GELU rather than SwiGLU, the improvement cannot be attributed exclusively to patching.

### Change 27 — imagenet — 2026-10-26

**Change:** Applied the same 8×8 patch embedding and spatial-mixing architecture to ImageNet32, using width 128, hidden dimension 512, depth 30, SwiGLU, dropout 0.1, AdamW, cosine scheduling, random cropping, horizontal flipping, and 20 epochs.
**Why:** Patch processing improved CIFAR-10 performance, so the next step was to evaluate whether the same architectural approach would benefit the more challenging ImageNet32 dataset.
**Expected:** Improved validation and test accuracy compared with the previous ImageNet32 best of 13.29% test accuracy, potentially with increased training time.
**Result:** Test accuracy improved from 13.29% to 15.14%. Best validation accuracy increased from 14.93% to 16.86%, and minimum validation loss decreased from 4.7963 to 4.6175. Training took 46.6 minutes. Validation accuracy continued improving through epoch 19, while validation loss plateaued near the end. This became the best ImageNet32 configuration tested.