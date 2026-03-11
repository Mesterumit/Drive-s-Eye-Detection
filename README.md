# Deep learning CPU development environment

A ready-to-use deep learning environment for CPU-only systems in VS Code. Includes both **PyTorch** and **TensorFlow** frameworks. Ideal for users without NVIDIA GPUs, including macOS users, or anyone who wants a lightweight environment for learning, prototyping, or CPU-based inference.

## What's included

| Category | Versions |
|----------|----------|
| **ML** | PyTorch 2.10, TensorFlow 2.16, Keras 3.3, Scikit-learn 1.4 |
| **Python** | Python 3.10, NumPy 1.24, Pandas 2.2, Matplotlib 3.10 |
| **Tools** | JupyterLab, TensorBoard |

> **Have an NVIDIA GPU?** Use the GPU version instead: [gperdrizet/deeplearning-GPU](https://github.com/gperdrizet/deeplearning-GPU)

## Project structure

```
deeplearning-CPU/
├── .devcontainer/
│   └── devcontainer.json       # Dev container configuration
├── data/                       # Store datasets here
├── logs/                       # TensorBoard logs
├── models/                     # Saved model files
├── notebooks/
│   ├── environment_test.ipynb  # Verify your setup
│   └── functions/              # Helper modules for notebooks
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

- **Docker** ([Windows](https://docs.docker.com/desktop/setup/install/windows-install) | [Mac](https://docs.docker.com/desktop/setup/install/mac-install) | [Linux](https://docs.docker.com/desktop/setup/install/linux))
- **VS Code** with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

> **Note:** This CPU-only container works on Windows, Linux, and both Intel and Apple Silicon Macs.

## Quick start

1. **Fork** this repository (click "Fork" button above)

2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/deeplearning-CPU.git
   ```

3. **Open VS Code**

4. **Open Folder in Container** from the VS Code command pallet (Ctrl+shift+p), start typing `Open Folder in`...

5. **Verify** by running `notebooks/environment_test.ipynb`

## Using as a template for new projects

You can use your fork as a template to quickly create new deep learning projects:

1. **Mark your fork as a template** (one-time setup):
   - Go to your fork on GitHub
   - Click **Settings**
   - Check the box for **Template repository**

2. **Create a new project** from your template:
   - Go to your fork on GitHub
   - Click the green **Use this template** button
   - Select **Create a new repository**
   - Enter your new project name and click **Create repository**

3. **Clone** your new project:
   ```bash
   git clone https://github.com/<your-username>/my-new-project.git
   cd my-new-project
   ```

4. **Clean up** (optional): Remove the example notebooks, then add your own code:
   ```bash
   rm -rf notebooks/*.ipynb
   git add -A && git commit -m "Initial project setup"
   git push
   ```

Now you have a fresh deep learning CPU project with the dev container configuration ready to go!

## Adding Python packages

### Using pip directly

Install packages in the container terminal:

```bash
pip install <package-name>
```

> **Note:** Packages installed this way will be lost when the container is rebuilt.

### Using requirements.txt (Recommended)

For persistent packages that survive container rebuilds:

1. **Create** a `requirements.txt` file in the repository root:
   ```
   scikit-image==0.22.0
   seaborn>=0.13.0
   plotly
   ```

2. **Update** `.devcontainer/devcontainer.json` to install packages on container creation by adding a `postCreateCommand`:
   ```json
   "postCreateCommand": "pip install -r requirements.txt"
   ```

3. **Rebuild** the container (`F1` → "Dev Containers: Rebuild Container")

Now your packages will be automatically installed whenever the container is created.

## TensorBoard

To launch TensorBoard:

1. Open the command palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run **Python: Launch TensorBoard**
3. Select the `logs/` directory when prompted

TensorBoard will open in a new tab within VS Code. Place your training logs in the `logs/` directory.

## Keeping your fork updated

```bash
# Add upstream (once)
git remote add upstream https://github.com/gperdrizet/deeplearning-CPU.git

# Sync
git fetch upstream
git merge upstream/main
```