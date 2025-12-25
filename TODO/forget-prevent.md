
**[System Role]**
You are a Senior Deep Learning Engineer specializing in Computer Vision and fine-tuning strategies for Vision Transformers (ViT).

**[Context]**
I am working on a **DINOv3 Fine-tune task for surgical scenes**.

* **Problem:** During fine-tuning, I am observing severe **"catastrophic forgetting,"** where the model loses the powerful semantic and geometric features learned during pre-training.

**[Task]**
Please help me modify my training code to implement two industry-standard "anti-forgetting" mechanisms. Implement the logic in a configurable/modular way.

**1. Implement "Feature Anchor Distillation"**

* **Core Logic:**
* Before training starts, create a copy of the Encoder, freeze its weights, and treat it as the **"Frozen Teacher"**, (which means i have two teachers now, one is the frozen one, another is the ema teacher.)
* During the training loop, pass the input image through both the "Frozen Teacher" and the "Student" (the Encoder currently being trained).
* Calculate the difference between their output feature maps (using **Cosine Similarity** or **Normalized L2 Distance**).
* Add this difference as a **regularization term** to the total loss function.


* **Goal:** Force the Student's feature space to remain statistically and spatially aligned with the pre-trained distribution while adapting to the new task.

**2. Implement "Layer-wise Learning Rate Decay" (LLRD)**

* **Core Logic:**
* Avoid using a uniform learning rate for the entire Encoder.
* Implement a **parameter grouping strategy** for the Optimizer.
* **Strategy:** The Decoder/Head should receive the **maximum base learning rate**. Within the Encoder (ViT), the learning rate should **decay layer-by-layer** from the top (output) down to the bottom (input).
* **Goal:** The bottom layers (handling edges/textures) should have a minimal learning rate to preserve general vision capabilities, while the top layers are allowed to adapt more aggressively to the specific domain.



**[Output Tips]**
* Ensure the code is modular and configurable and easy to integrate into an existing PyTorch training script.
* The modification may mainly in /bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/dinov3/dinov3/train/ssl_meta_arch.py and /bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/dinov3/dinov3/train/train.py.