from collections import OrderedDict
from types import SimpleNamespace

cfg_dict = {
    "repo_dir": "/bd_byta6000i0/users/surgicaldinov2/kyyang/dinov3",
    "hub_name": "dinov3_vitb16",
    "pretrained_weights_path": "/bd_byta6000i0/users/surgicaldinov2/kyyang/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "trained_checkpoint": "/bd_byta6000i0/users/surgicaldinov2/kyyang/dinov3/results/1/eval/training_3999/teacher_checkpoint.pth",
}
cfg = SimpleNamespace(**cfg_dict)


import torch.hub
import torch

encoder = torch.hub.load(
    cfg.repo_dir, cfg.hub_name, source="local", weights=cfg.pretrained_weights_path
)

state = torch.load(cfg.trained_checkpoint, map_location="cpu")
teacher = state.get("teacher")
if teacher is None:
    raise KeyError("trained_checkpoint missing 'teacher' key")
prefix = "backbone."
new_state = OrderedDict(
    (k[len(prefix) :] if k.startswith(prefix) else k, v) for k, v in teacher.items()
)
missing, unexpected = encoder.load_state_dict(new_state, strict=False)

inputs = torch.randn(3, 3, 224, 224)

outputs = encoder.get_intermediate_layers(inputs, n = [5, 7, 9, 11])

outputs[0].shape # (3, 196, 768)
