# Training on the desktop

How to run `exp-tuning.ipynb` on this PC's RTX 4080 SUPER. You don't need to type Linux commands: everything
below happens in VS Code.

**Why Linux:** TensorFlow can only use an NVIDIA GPU on Linux; its Windows GPU support ended with version 2.10.
WSL2 is Microsoft's built-in Linux and runs inside Windows like an app. VS Code connects to it, so the notebook
looks and works as before.

## What's installed (2026-10-03)

| | |
|---|---|
| Linux | Ubuntu 24.04 in WSL2, user `sheng` |
| Python environment | `~/venvs/fer`, the VS Code kernel **Python (fer)**: Python 3.12, TensorFlow 2.19.1 (CUDA 12.5, cuDNN 9.3), tf-keras 2.19.0, transformers 4.54.1 |
| GPU | RTX 4080 SUPER; TensorFlow can use 13.5 of its 16 GB |
| Datasets | Already in `data/`: 40,969 training and 10,245 test images (FER2013 plus RAF-DB), plus the Kaggle baseline's final weights in `data/baseline_run/`. `data/emotion-detection-fer/` is no longer used: it's FER2013's test images again |
| Downloaded once | VGG16 and ViT weights; MediaPipe and its face model (in `~/fer_scratch/landmark_pkgs`) |

## 1. Open the project in Linux mode
1. In VS Code: **File → Open Recent → ultrasound_Model [WSL: Ubuntu-24.04]**.
2. Check the bottom-left corner. It must say **WSL: Ubuntu-24.04**; without it, the notebook runs on Windows and
   can't use the GPU.
3. If VS Code asks whether you trust the folder, choose **Yes, I trust the authors**.

Not in Open Recent? Click the `><` button in the bottom-left corner, choose **Connect to WSL**, then
**File → Open Folder** → `/mnt/c/Users/sheng/Documents/GitHub/ultrasound_Model`.

That is the same folder as `C:\Users\sheng\Documents\GitHub\ultrasound_Model`; Linux calls the C: drive
`/mnt/c`. There is only one copy, so File Explorer and GitHub Desktop keep working as usual.

## 2. Pick the kernel (first time only)
Open `exp-tuning.ipynb`, click the kernel name at the top right (or **Select Kernel**), then choose
**Python Environments… → fer** (`~/venvs/fer/bin/python`). VS Code remembers the choice.

Don't pick the plain **Python 3.12.3** (`/usr/bin/python3`). That's Ubuntu's own Python, without TensorFlow.
Both show version 3.12.3, so go by the name **fer** and the path. If VS Code asks you to install `ipykernel`, it
has picked the wrong one: close the prompt and pick fer. If fer isn't listed, press Ctrl+Shift+P, run
**Developer: Reload Window**, and try again.

## 3. Run the two arms
The experiment compares the image-only model (**baseline**) with the same model plus 34 face measurements
(**geometry**). For each run, set two lines in the settings cell (the one starting `# Experiment settings and
paths`), then click **Run All** at the top of the notebook.

**Run 1: baseline, evaluation only (about 30–45 min).** This reuses your finished Kaggle run:
```python
USE_GEOMETRY = False
PRETRAINED_WEIGHTS = os.path.join("data", "baseline_run", "vgg16_cbam_vit_incremental_unfreeze_improved")
```

**Run 2: geometry, trained from scratch (about 8.5 hours; measured 2026-10-03).**
```python
USE_GEOMETRY = True
PRETRAINED_WEIGHTS = None
```
Before run 2, click **Restart** in the notebook toolbar, then **Run All**. Restarting frees the GPU memory that
run 1 used.

The last cell compares the two arms:
- After run 1, it only reports that the geometry predictions are missing.
- After run 2, it prints the comparison table and a plot.

Results are saved in `outputs/` (also visible in File Explorer):
- `predictions_<arm>_test.npz`: the per-image predictions
- `test_near_duplicates.csv`: the 604 test images that have a near-identical copy in the training data. The
  headline results leave them out.
- `vgg16_cbam_vit_geometry.*`: the trained geometry model
- `landmarks/`: the face measurements, computed once and reused

Optional, for the cleanest comparison: also retrain the baseline here (`PRETRAINED_WEIGHTS = None`), so that both
arms ran on the same machine. It adds another 8–9 hours.

### Cross-dataset runs: train on RAF-DB only, test on FER2013 (two runs, about 3–4 hours each)
The models train only on RAF-DB's 12,271 training images. RAF-DB and FER2013 test images are both scored. The
files are named `..._train-RAF-DB`, so they don't replace your two main runs.

**Run A: baseline.** Click **Restart**, set these lines in the settings cell, then click **Run All**:
```python
USE_GEOMETRY = False
TRAIN_ON = "RAF-DB"
PRETRAINED_WEIGHTS = None
```
After training, the **Test results by subset** lines give the two numbers that matter:
- **RAF-DB:** the dataset it trained on.
- **FER2013 (not trained on):** the cross-dataset test.

The comparison at the end only says the geometry run is missing.

**Run B: geometry.** Click **Restart**, change only `USE_GEOMETRY = True`, then click **Run All**. The face
landmarks for RAF-DB are already computed.
- The comparison at the end now compares the two RAF-DB-trained runs.
- Its "FER2013 … (not trained on)" table shows whether geometry helps across datasets.

Afterwards, set `TRAIN_ON = "both"` again for normal runs.

## 4. While it runs
**Normal output you can ignore:**
- Red TensorFlow startup lines such as `Unable to register cuDNN factory` or `computation placer already
  registered`.
- `TensorFlow and JAX classes are deprecated…`: a notice from transformers. The pinned version still works.
- Several quiet minutes at the start, while about 150,000 images are rebuilt.

**Reading the epoch lines.** An epoch is one pass over the roughly 58,000 training images. Training runs up to
25 epochs with the ViT frozen, then up to 35 with everything trainable. It stops early once the validation loss
stops improving. Each epoch shows a progress bar with an ETA and ends with a summary line like this (the numbers
are made up):
```
Epoch 1/25
3650/3650 [==============================] - 420s 115ms/step - loss: 0.16 - accuracy: 0.35 - val_loss: 0.12 - val_accuracy: 0.54
```
- `420s` is that epoch's time.
- Measured on this PC (2026-10-03 geometry run): about 10.5 min per epoch in phase 1 and 19.5 min in phase 2.
  On Kaggle a phase-2 epoch took 62 min.
- **If the ETA for one epoch is in hours, the GPU isn't being used.** Stop the run with ■ **Interrupt** and ask
  Claude.
- To see the GPU working, open Task Manager → Performance → GPU. "Dedicated GPU memory" sits at about 13–14 GB
  during training.

**Keep it running:**
- Leave VS Code open. Closing its window stops the run. The screen may turn off; the PC is set to never sleep.
- Windows Update can restart the PC overnight. Before a long run, go to Settings → Windows Update →
  **Pause updates**.
- To stop a run on purpose, click ■ **Interrupt** in the notebook toolbar.

## 5. If something goes wrong

| You see | Do this |
|---|---|
| `WARNING: no GPU found`, or the bottom-left corner doesn't say WSL | Make sure you opened the WSL entry (step 1). If it still happens, close VS Code, run `wsl --shutdown` in PowerShell, then reopen the project from Open Recent. |
| `Killed`, or "The kernel died" | Linux ran out of memory: WSL gets about 15 GB of the PC's 31 GB by default. Ask Claude to raise the limit; it creates `C:\Users\sheng\.wslconfig` with `memory=24GB` and restarts WSL. |
| `ResourceExhaustedError` (out of GPU memory) | Click **Restart**, then **Run All**. Don't run two notebooks at once. |
| `transformers …: needs <4.55` | Run the install cell (the second cell), click **Restart**, then **Run All**. |
| Anything else | Copy the last 20 lines of the error to Claude. |

## Where things are
- **Project, data and outputs:** `C:\Users\sheng\Documents\GitHub\ultrasound_Model`; from Linux,
  `/mnt/c/Users/sheng/Documents/GitHub/ultrasound_Model`. Git and GitHub Desktop work from Windows as before.
  `data/` and `outputs/` are in `.gitignore`.
- **Linux-side files:** File Explorer sidebar → **Linux** → Ubuntu-24.04 → `home/sheng`:
  - `fer_scratch/`: the images rebuilt on every run (about 4 GB) and MediaPipe. Safe to delete between runs; the
    notebook recreates it and downloads MediaPipe again.
  - `venvs/fer/`: the Python environment.

## How it was set up
For reference, in case it ever needs redoing:
1. In PowerShell as administrator, run `wsl --install --no-distribution`, then restart. Then run
   `wsl --install -d Ubuntu-24.04` and create the Linux user.
2. As root, run `apt-get install python3-venv python3-pip libgles2`. MediaPipe needs `libgles2`; `libgl1`,
   `libegl1` and `git` come with Ubuntu 24.04.
3. Create `~/venvs/fer` with `python3 -m venv`, then pip-install the list from the notebook's Requirements cell.
   Then run `python -m ipykernel install --user --name fer --display-name "Python (fer)"`.
4. In the repo, run `git config --local core.autocrlf true`, so Linux git and Windows git agree on line endings.
5. In VS Code, install the WSL extension on Windows, and the Python and Jupyter extensions inside WSL.
6. VS Code doesn't look in `~/venvs` for environments by default. The WSL-side settings file
   `~/.vscode-server/data/Machine/settings.json` therefore sets `"python-envs.globalSearchPaths": ["/home/sheng/venvs"]`
   and `"python.defaultInterpreterPath": "/home/sheng/venvs/fer/bin/python"`.
