"""Hugging Face Hub checkpointing for training on ephemeral compute (e.g. Colab).

Saves and resumes training checkpoints to a *private, repo-scoped* HF model repo,
so a dropped session never costs more than one epoch and no Google Drive access
(or any broad credential) is required.

Usage:
    from hf_checkpoint import HFCheckpoint
    ckpt = HFCheckpoint(repo_id="you/imagenet-proxy-ckpts", token=HF_TOKEN)

    state = ckpt.load("last.pt", map_location=device)   # None on first run
    ...
    ckpt.save({"epoch": epoch, "model": model.state_dict(), ...}, "last.pt")
"""

import os
import torch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


class HFCheckpoint:
    """Save/load PyTorch checkpoints to a Hugging Face Hub repo."""

    def __init__(self, repo_id, token, repo_type="model", local_dir="/content"):
        """
        Args:
            repo_id:   "username/repo-name" of an existing (private) HF repo.
            token:     A fine-grained token with read+write to *only* this repo.
            repo_type: Usually "model".
            local_dir: Scratch dir for staging files before upload.
        """
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.local_dir = local_dir
        self.token = token
        self.api = HfApi(token=token)
        os.makedirs(local_dir, exist_ok=True)

    def save(self, state, filename="last.pt", commit_message=None):
        """Serialize `state` (a dict) and upload it to the Hub.

        Overwrites `filename` in the repo each call (one commit per save).
        Returns the local staging path.
        """
        local_path = os.path.join(self.local_dir, filename)
        torch.save(state, local_path)
        self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            commit_message=commit_message or f"checkpoint: {filename}",
        )
        return local_path

    def load(self, filename="last.pt", map_location="cpu"):
        """Download and load `filename` from the Hub.

        Returns the loaded checkpoint dict, or None if the file does not exist
        yet (i.e. the first run). Pass map_location=device so model AND optimizer
        tensors land on the right device and resume cleanly.

        weights_only=False is safe here: the file comes from your own private repo.
        """
        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type=self.repo_type,
                token=self.token,
            )
        except EntryNotFoundError:
            return None
        return torch.load(path, map_location=map_location, weights_only=False)
