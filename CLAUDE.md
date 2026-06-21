# ImageNet-proxy project

- model.py holds the architecture. Always keep build_model(num_classes) -> nn.Module;
  train.py depends on that signature.
- Input is 3×160×160; output must be (batch, num_classes) logits.
- After editing model.py, run `python check_model.py` to verify shapes before committing.
- Training runs on Colab via train.py (Imagenette proxy); checkpoints go to a private HF repo.
