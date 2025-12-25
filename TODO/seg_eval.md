
## Refined Prompt for the AI Agent

**Task:** Create a complete PyTorch Lightning pipeline to evaluate a **DINOv3** backbone on the **CholecSeg8k** dataset for semantic segmentation.

### 1. Project Requirements & Frameworks

* **Frameworks:** PyTorch Lightning, Weights & Biases (W&B) for logging, and Hydra for configuration.
* **Backbone:** Use a **DINOv3** backbone. It must be **frozen** (`requires_grad=False`).
* **Structure:** Organize the code into a `LightningModule` for the model/loss, a `LightningDataModule` for the dataset, and a main execution script.

### 2. Dataset: CholecSeg8k (Video-Level Split)

* **Paths:** * Images: `/bd_byta6000i0/users/dataset/MedicalImage/CholecSeg8k/raw/videoXX/videoXX_XXXXX/frame_XX_endo.png`
* Masks: `/bd_byta6000i0/users/dataset/MedicalImage/CholecSeg8k/raw/videoXX/videoXX_XXXXX/frame_XX_endo_color_mask.png`


* **Splitting Logic:** You **must** split the dataset at the **video level**. Ensure that all frames from a specific video directory are either entirely in the training set or entirely in the validation set to prevent data leakage.
* **Color Mapping:**  the color-to-class mapping logic from `eval/segmentation/colormap.py`.

### 3. Segmentation Head & Loss

**Do not import these from external local paths.** Re-implement the following logic directly or copy the code into the generated code:

Build the head and loss based on the reference architecture used in `/bd_byta6000i0/users/surgicaldinov2/kyyang/surgicaldinov3/dinov3/dinov3/eval/segmentation`.

### 4. Configuration (Single-File Hydra YAML)

* **Format:** Provide the Hydra configuration **only in a single YAML file**.
* **Constraint:** Do not split the config into multiple files (no `conf/db`, `conf/model`, etc.) and **do not use Python dataclasses**.

### 5. Implementation Details

* **Self-Contained:** The generated code should be ready to run with minimal external dependencies beyond standard ML libraries.
* **Metrics:** Log **Mean IoU (mIoU)** and **Global Accuracy** to W&B during the validation step.
* **DINOv3 Integration:** Ensure the forward pass correctly handles the DINOv3 output (e.g., extracting the correct feature tokens for the segmentation head).
* Make sure the code can be run from the eval/segmentation directory.
* make sure the wandb, hydra and lightning logs are in the structure logs/run/wandb, logs/hydra, logs/lightning
* Make sure the run_name is in the config, and the run or logs name should include the env variable `OAR_JOB_ID`. and the wandb_run_name should be in the config.

---

### Key Improvements Made:

* **Single-File Constraint:** Added a specific instruction to keep the Hydra YAML in one file as requested.
* **Head/Loss Context:** Added clear instructions to re-implement the internal logic rather than importing it, ensuring the script is portable.
* **Data Leakage Prevention:** Explicitly emphasized the video-level split, which is critical for medical imaging tasks like CholecSeg8k.
