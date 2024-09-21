# News Research Tool

Welcome to the **News Research Tool**! This application allows users to explore and analyze news articles from various sources.

## Configuration

Spaces are configured through the YAML block at the top of the `README.md` file at the root of the repository. Below are the accepted parameters:


title: "News Research Tool"  # Display title for the Space.
emoji: "📰"                  # Space emoji (emoji-only character allowed).
colorFrom: "blue"            # Color for Thumbnail gradient (e.g., red, yellow, green).
colorTo: "green"             # Color for Thumbnail gradient (e.g., red, yellow, green).
sdk: "streamlit"             # Can be either gradio, streamlit, docker, or static.
python_version: "3.10"       # Any valid Python 3.x or 3.x.x version. Defaults to 3.10.
sdk_version: "0.84.0"        # Specify the version of the selected SDK.
suggested_hardware: "gpu-medium"  # Suggested hardware for running the Space.
suggested_storage: "medium"   # Suggested permanent storage for the Space.
app_file: "app/app.py"       # Path to your main application file.
app_port: 7860                # Port on which your application is running (used if sdk is docker).
base_path: "/"                # Initial URL to render.
fullWidth: true               # Rendered inside a full-width column.
header: "default"             # Can be either mini or default.
short_description: "A tool to analyze news articles."  # Short description of the Space.
models:                        # HF model IDs used in the Space.
  - "openai-community/gpt2"
datasets:                     # HF dataset IDs used in the Space.
  - "mozilla-foundation/common_voice_13_0"
tags:                         # List of terms that describe your Space.
  - "news"
  - "research"
thumbnail: "url/to/thumbnail.png"  # URL for a custom thumbnail for social sharing.
pinned: false                 # Whether the Space stays on top of your profile.
hf_oauth: false               # Whether a connected OAuth app is associated to this Space.
hf_oauth_scopes:              # Authorized scopes of the connected OAuth app.
  - "openid"
  - "profile"
hf_oauth_expiration_minutes:  480  # Duration of the OAuth token in minutes.
disable_embedding: false       # Whether the Space iframe can be embedded in other websites.
startup_duration_timeout: "30m" # Custom startup duration timeout for your Space.
custom_headers:               # Custom HTTP headers.
  cross-origin-embedder-policy: "require-corp"
preload_from_hub:             # List of Hugging Face Hub models or files to preload.
  - "warp-ai/wuerstchen-prior text_encoder/model.safetensors,prior/diffusion_pytorch_model.safetensors"
