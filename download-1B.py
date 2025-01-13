from huggingface_hub import snapshot_download

hf_token = str(input("Insert hugging face token: "))

snapshot_download(
    repo_id="allenai/OLMo-1B-0724-hf",
    token=hf_token,
    local_dir="models/base/OLMo-1B-0724-hf",
)
# Downloading 1B model
snapshot_download(
    repo_id="llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning",
    token=hf_token,
    local_dir="models/base/olmo-1B-model-semeval25-unlearning",
)
