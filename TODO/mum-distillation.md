
**Role:** You are a Senior AI Engineer familiar with the source code of **DINOv3/DINOv2** and the **MuM (Multi-View Masked Image Modeling)** repository.

**Context:** I have the full codebase for **MuM**, which already contains implementations for:

1. The `MuMDecoder` (with Axial RoPE and Multi-view Attention).
2. Positional embedding helpers.
3. Masking generators.
I also have a standard **DINO** training script.

**Objective:** Modify the existing DINO training pipeline to add **MuM's pixel reconstruction objective** as an auxiliary task. The goal is to train the Shared Encoder to have both semantic understanding (DINO loss) and 3D spatial awareness (MuM loss).

**Task:** Please guide me or generate code snippets to implement the following three critical modifications without inventing specific variable names (infer them from context or use generic placeholders):

### 1. Hybrid Data Loading (The "Two-Stream" Batch)

Modify the data loading logic (Dataset & Collator) to yield a **hybrid batch** for every iteration:

* **Stream A (for DINO):** Load a standard single image (or global crops) with **Strong Augmentation** (Color Jitter, etc.).
* **Stream B (for MuM):** For the *same* scene/sample, load a **Sequence/Video Clip** with a random length (sampled within a specific range, e.g., 2 to N frames).
* Apply **Weak Augmentation** (Resize + Sync Flip) to this clip.
* Ensure the geometric relationship between frames in the clip is preserved (no independent distortions).


* **Output:** The batch should contain both the strongly augmented view (for the Teacher/Student DINO head) and the weakly augmented sequence (for the MuM Decoder).

### 2. The "Bridge" Logic (Filter & Select)

In the training loop's `forward` pass, implement the logic to bridge the DINO Encoder and the MuM Decoder.

* **Input:** Pass the masked sequence (Stream B) into the DINO Encoder.
* **Constraint:** The DINO Encoder uses a "replace-with-mask-token" strategy (keeps sequence length constant). The MuM Decoder expects a "dropped-token" input (sparse sequence).
* **Action:** Implement a filtering operation on the Encoder's output:
1. Identify indices of tokens that were **NOT** masked (visible tokens).
2. **Gather/Select** only these visible tokens from the Encoder's output features.
3. Prepare the auxiliary tensors required by the MuM Decoder (e.g., indices to restore the original order), utilizing the existing utilities in the MuM repo if available.



### 3. Decoder Integration & Loss Calculation

Feed the filtered tokens into the pre-existing MuM Decoder instance.

* **Forward:** Pass the sparse visible tokens and the restore indices into the Decoder.
* **Loss:** Calculate the MSE loss between the Decoder's predicted pixels and the normalized ground-truth pixels of the masked patches.
* **Optimization:** Combine this MuM Loss with the existing DINO Loss using a weighting factor and backpropagate.

**Requirement:**

* Do not write the `Decoder` or `RoPE` classes from scratch; assume I will import them from the MuM codebase.
* Focus on the **glue code** inside the training loop and the **data collation logic**.
* Ensure the gradients flow back through the Shared Encoder from both heads.
