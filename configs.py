from huggingface_hub import snapshot_download

my_local_config_path = snapshot_download(
    repo_id="stabilityai/stable-diffusion-3-medium-diffusers",
    allow_patterns=["*.model", "**/*.model"]
)