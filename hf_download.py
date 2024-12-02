from huggingface_hub import snapshot_download, login

login()

snapshot_download(repo_id = "meta-llama/Llama-3.1-8B",
                  ignore_patterns = ["original", "*.pth"],
                  local_dir = r"C:\home\bizon\VLLM\Llama-3.1-8B")