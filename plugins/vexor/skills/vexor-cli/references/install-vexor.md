# Install Vexor

Use this when the `vexor` command is missing.

## Steps

1. Verify if it is already installed:
   ```bash
   vexor --help
   ```

2. Install with pip (recommended):
   ```bash
   python -m pip install -U vexor
   ```

3. (Optional) Install with pipx for isolation:
   ```bash
   pipx install vexor
   ```

4. Verify the install:
   ```bash
   vexor --help
   ```

## If the command is still missing

- Try running via module:
  ```bash
  python -m vexor --help
  ```
- Ensure the Python scripts directory is on PATH (e.g., `~/.local/bin` for user installs).

## After install: configure a provider

Vexor needs a provider configuration before semantic search works.

- **Remote providers (openai/gemini/custom)**: set an API key.
  ```bash
  vexor config --set-api-key "<YOUR_KEY>"
  ```
  Optional: choose provider/model.
  ```bash
  vexor config --set-provider openai
  vexor config --set-model text-embedding-3-small
  ```

- **Local provider** (no API key, but requires local model setup):
  ```bash
  pip install "vexor[local]" or pip install "vexor[local-cuda]"
  vexor local --setup --model intfloat/multilingual-e5-small [--cuda]
  ```

If you cannot provide credentials or choose a local model, ask the user to assist.
