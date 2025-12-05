   DA3 Backbone MIM Distillation – Integrated Spec / TODO

   Original intent
   •  Use da3_backbone as a teacher in a MIM-style distillation setup.
   •  Teacher (DA3) sees the full, clean image; student sees the masked image.
   •  Supervision pipeline:
     •  Use da3_backbone on collated_global_crops_clean to obtain supervising tokens.
     •  Take masked_patches_pre_head from the student backbone and pass through a small MLP with GELU (projector)
        into the DA3 feature space.
     •  Compute cosine-similarity loss between DA3 tokens and projected student tokens.
     •  Enforce both teacher and student tokens to be L2-normalized (norm = 1).

   1) Data Flow (using collate.py)
   •  File: dinov3/dinov3/data/collate.py.
   •  collate_data_and_cast already returns:
     •  collated_global_crops: stacked from sample["global_crops"], shape [n_global_crops * B, 3, H, W].
     •  collated_global_crops_clean: stacked from sample["global_crops_clean"] with identical indexing, so same
        shape and ordering as collated_global_crops.
     •  collated_local_crops, collated_masks, mask_indices_list, masks_weight, upperbound, n_masked_patches.
   •  Spec decisions:
     •  Do not modify collate_data_and_cast: collated_global_crops_clean is already produced and cast to the common
        dtype.
     •  In SSLMetaArch.forward_backward, when DA3 distillation is enabled, we read
        data["collated_global_crops_clean"] and move it to CUDA (and to float32 inside the DA3 teacher forward
        helper).
     •  Later, if get_batch_subset is used in your training pipeline and DA3 is enabled, extend it to subset
        collated_global_crops_clean in parallel with collated_global_crops so the DA3 supervision stays consistent.

   2) Config Changes (update vitb_full_config.yaml first)
   •  File: dinov3/dinov3/configs/vitb_full_config.yaml.
   •  Add a new top-level block for DA3 MIM distillation:

   yaml
       da3:
         enabled: false              # Turn on DA3 MIM distillation
         loss_weight: 1.0            # Global weight for cosine distillation loss
         embed_dim: 1024             # DA3 token dimension (must match da3_backbone output)
         projector_hidden_dim: 2048  # Hidden dim of student projector MLP
         ckpt_path: ""              # Path to DA3 teacher checkpoint
         use_fsdp: false             # Wrap DA3 teacher as FSDP inference-only model if true

   •  Notes:
     •  enabled=false keeps the default behavior identical to existing DINOv3/iBOT training.
     •  embed_dim will size the projector’s output; it must match the DA3 supervision token dimensionality.
     •  projector_hidden_dim is chosen to mirror other heads (DINO/IBOT); can be tuned later.
     •  ckpt_path will be used inside build_da3_backbone or immediately after instantiation to load DA3 weights.
     •  use_fsdp controls whether DA3 teacher participates in FSDP as an inference-only model (see section 6).

   3) DA3 Teacher Backbone Integration
   •  Create a Python wrapper from utility/da3_backbone.ipynb, e.g. utility/da3_backbone.py with:

   python
       def build_da3_backbone(da3_cfg):
           """Return (model, embed_dim) for DA3 teacher.

           model: nn.Module taking [N, 3, H, W] -> [N, P_da3, D_da3].
           embed_dim: D_da3.
           """

   •  Responsibilities of build_da3_backbone:
     •  Construct the DA3 model according to da3_cfg.
     •  Load weights from da3_cfg.ckpt_path if non-empty.
     •  Move model to the appropriate device/dtype as needed by the caller.
     •  Return (model, embed_dim) where embed_dim is the DA3 token dimension.
   •  In SSLMetaArch.__init__ (file: dinov3/dinov3/train/ssl_meta_arch.py):
     •  Read cfg.da3.
     •  If cfg.da3.enabled:
       •  Build DA3 teacher:

         ```python
         from utility.da3_backbone import build_da3_backbone
         self.da3_teacher, self.da3_embed_dim = build_da3_backbone(self.cfg.da3)
         ```

   •  Assert consistency with config:

         ```python
         assert self.da3_embed_dim == self.cfg.da3.embed_dim
         ```

   •  Freeze and set eval:

         ```python
         self.da3_teacher.requires_grad_(False)
         self.da3_teacher.eval()
         ```

   •  Optionally move to the right device here or rely on distributed/FSDP setup.

   4) Student Projector from masked_patches_pre_head to DA3 Space
   •  Still in SSLMetaArch.__init__, if cfg.da3.enabled:
     •  Define projector MLP with GELU on the student backbone embed dim self.embed_dim (same D as CLS/patch
        tokens):

       ```python
       self.da3_projector = nn.Sequential(
           nn.Linear(self.embed_dim, self.cfg.da3.projector_hidden_dim),
           nn.GELU(),
           nn.Linear(self.cfg.da3.projector_hidden_dim, self.cfg.da3.embed_dim),
       )
       ```

   •  Optional: register under student_model_dict["da3_projector"] so it goes through the same param-group/FSDP
      path as other student heads.
   •  In init_weights():
     •  When cfg.da3.enabled, call self.da3_projector.apply(init_fn) or use the same initialization scheme you
        apply to DINO/IBOT heads.
   •  Projector input at training time:
     •  Use student_global["masked_patch_pre_head"] from get_student_output, shaped [n_masked_patches, 
        self.embed_dim].

   5) DA3 Teacher Forward on collated_global_crops_clean
   •  Add helper to SSLMetaArch:

   python
       @torch.no_grad()
       def get_da3_teacher_output(self, clean_global_crops, mask_indices_list):
           """Compute DA3 teacher tokens for masked positions.

           clean_global_crops: [n_global_crops, B, 3, H, W]
           mask_indices_list: [n_masked_patches], indices into flattened [n_global_crops * B * P].
           """

   •  Implementation outline:
     •  Flatten crops and cast to float32 for DA3 teacher stability:

       ```python
       n_crops, B, C, H, W = clean_global_crops.shape
       images = clean_global_crops.flatten(0, 1).float()  # [n_crops * B, C, H, W]
       da3_tokens = self.da3_teacher(images)              # [n_crops * B, P_da3, D_da3]
       ```

   •  If DA3 patch grid differs from the student’s grid:
     •  Optionally resize (interpolate) DA3 spatial features to match the student’s patch-grid resolution before
        flattening to [N, P_student, D_da3]. (Keep this flexible in case the DA3 backbone has different strides.)
   •  Flatten over crops and patches:

       ```python
       da3_patches = da3_tokens.flatten(0, 1)  # [n_crops  B  P, D_da3]
       ```

   •  Select only masked positions using the same mask_indices_list used for masked_patches_pre_head:

       ```python
       da3_masked = torch.index_select(da3_patches, dim=0, index=mask_indices_list)
       ```

   •  Return:

       ```python
       return {"masked_da3_tokens": da3_masked}
       ```

   6) Wiring into forward_backward and Loss Computation
   •  In SSLMetaArch.forward_backward:
     •  After reading collated_global_crops / collated_local_crops and masks, also read clean crops when
        cfg.da3.enabled:

       ```python
       if self.cfg.da3.enabled:
           clean_global_crops = data["collated_global_crops_clean"].cuda(non_blocking=True)
       else:
           clean_global_crops = None
       ```

   •  After computing teacher_global and (student_global, student_local):
     •  If DA3 is enabled, compute DA3 teacher output:

         ```python
         if self.cfg.da3.enabled:
             n_global_crops = student_global["cls_after_head"].shape[0]
             B = data["collated_local_crops"].shape[0] // self.n_local_crops
             da3_global = self.get_da3_teacher_output(
                 clean_global_crops.unflatten(0, (n_global_crops, B)),
                 mask_indices_list=mask_indices_list,
             )
         else:
             da3_global = None
         ```

   •  Pass da3_global as an additional argument into compute_losses.

   •  In compute_losses:
     •  Extend signature to accept da3_global (can be None when disabled).
     •  After DINO, KoLeo, iBOT, and Gram losses, add DA3 MIM loss when enabled:

       ```python
       if self.cfg.da3.enabled and da3_global is not None:
           student_masked = student_global["masked_patch_pre_head"]  # [n_masked_patches, D_student]
           student_proj = self.da3_projector(student_masked)          # [n_masked_patches, D_da3]

           teacher_tokens = da3_global["masked_da3_tokens"]          # [n_masked_patches, D_da3]

           student_norm = F.normalize(student_proj, dim=-1)
           teacher_norm = F.normalize(teacher_tokens, dim=-1)

           cos_sim = (student_norm * teacher_norm).sum(dim=-1)
           mim_loss = 1.0 - cos_sim.mean()

           loss_dict["da3_mim_loss"] = mim_loss
           loss_accumulator += self.cfg.da3.loss_weight * mim_loss
       ```

   •  This enforces both token sets to have norm = 1 via F.normalize, and uses 1 - cos(theta) as the per-token
      loss.

   7) Distributed / FSDP and Param Groups
   •  Projector params:
     •  If da3_projector is attached to student_model_dict, it will automatically be included in get_params_groups
        and wrapped in FSDP like other student components.
   •  DA3 teacher handling:
     •  If cfg.da3.use_fsdp is true:
       •  In prepare_for_distributed_training, add self.da3_teacher to inference_only_models and choose a process
          group, typically the default one (mirroring distillation/Gram teachers).
     •  If use_fsdp is false:
       •  Keep DA3 teacher as a plain nn.Module on each rank, moved to CUDA and always used under torch.no_grad().
   •  Data/broadcast consistency:
     •  collated_global_crops_clean is built with the same crop order as collated_global_crops; mask_indices_list
        indexes the flattened patch tokens for each crop. Using the same index tensor on both student and DA3
        teacher ensures we supervise exactly the masked student patches.

   8) Practical TODO List (for TODO/da3-distillation.md)
   [ ] Convert utility/da3_backbone.ipynb into dinov3/dinov3/utils/da3_backbone.py with build_da3_backbone(da3_cfg) returning
       (model, embed_dim).
   [ ] Update dinov3/dinov3/configs/vitb_full_config.yaml to include the da3 block (with reasonable defaults and
       enabled: false).
   [ ] In SSLMetaArch.__init__:
     [ ] Instantiate self.da3_teacher and freeze it when cfg.da3.enabled.
     [ ] Instantiate self.da3_projector (MLP + GELU) and register it, preferably under student_model_dict.
   [ ] In SSLMetaArch.init_weights():
     [ ] Initialize self.da3_projector alongside other heads.
   [ ] Add get_da3_teacher_output method to SSLMetaArch using collated_global_crops_clean and mask_indices_list.
   [ ] In forward_backward:
     [ ] Read collated_global_crops_clean when DA3 is enabled and call get_da3_teacher_output.
     [ ] Pass da3_global into compute_losses.
   [ ] In compute_losses:
     [ ] Add DA3 cosine MIM loss on L2-normalized tokens and accumulate into loss_accumulator with
         cfg.da3.loss_weight.
   [ ] In prepare_for_distributed_training:
     [ ] Optionally register da3_teacher as an inference-only model when cfg.da3.use_fsdp is true.
   [ ] Add sanity checks/logging (shapes, cosine similarity stats) to validate that DA3 supervision lines up with
       student masks.


**9) Sanity Checks and Experiments**
- Add quick assertions/logging:
  - Check shapes of `student_masked`, `teacher_masked`, and `mask_indices_list` match expectations.
  - Log mean cosine similarity before training for a small batch to confirm the loss is finite and gradients flow.