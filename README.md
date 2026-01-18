# KRYPTO-project

## Report
Instrukcja do korzystania z latexa lokalnie w VSCode: https://medium.com/@erencanbulut/boost-your-latex-workflow-with-vs-code-and-github-f346b74677be

## Implementation

### Setup

1. Install uv:
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync dependencies:**
   ```bash
   uv sync
   ```
   This will:
   - Create a virtual environment (`.venv`)
   - Install all dependencies from `pyproject.toml`
   - Use exact versions from `uv.lock`

3. **Use the virtual environment to run scripts:**
   ```bash
   uv run python src/main.py
   ```

### Useful commands

- Add a new dependency:
  ```bash
  uv add <package-name>
  ```

