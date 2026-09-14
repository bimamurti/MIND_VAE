
# MIND — Continual Learning for Crowd Trajectory Prediction
Detailed Result And Appendix read here: [ICRA-27 appendix.pdf](<ICRA27-supplement appendix.pdf>)

Overview
--------
MIND is a coreset selection strategy for continual learning applied to crowd trajectory prediction. It combines Membership, Influence, and Difficulty criteria to select representative samples (the "MIND" strategy) and addresses the stability–plasticity trade-off better than several baseline coreset methods.
![Pipeline overview](main%20pipeline.png)

Requirements (tested)
---------------------
- Python 3.9+
- PyTorch (tested with 2.8.0+cu126; newer CUDA versions should work)
- torchvision (compatible with chosen PyTorch)
- numpy, matplotlib, tqdm, tensorboard

Install
-------
1. Create and activate a virtual environment (recommended):

    python -m venv .venv
    source .venv/bin/activate

2. Install dependencies:

    pip install -r requirements.txt

Configuration
-------------
Before running experiments, set the project paths in `ContinualLearning/mainContinual.py` (or export them as environment variables). Example variables to set inside the file:

- `homefolder` — project root (e.g. `/home/user/projects/MIND_VAE/`)
- `datafolder` — dataset location
- `ckptpath` — directory to save checkpoints and logs
- `configpath` — path to the config file to use (e.g. `config/mot.py`)

Training (quick start)
----------------------
Open `mainContinual.py` and in the `main()` function uncomment the training method you want to run. Available entry points (examples):

- `independentlearning()` — independent training baseline
- `noreplystrategy()` — zero-shot method
- `exreplay()` — experience replay baseline
- `experienceReplay(20)` — uniform replay with buffer size 20
- `BCSRexperienceReplay()` — BCSR replay method
- `RELexperienceReplay()` — REL replay method
- `ProposedexperienceReplay()` — MIND (proposed)

Then run:

    python mainContinual.py

Evaluation
----------
To evaluate a trained model, set `EVAL_CKPTPATH` in `mainContinual.py` to point to the checkpoint directory (typically under `log_eth/`) and enable the evaluation call in that script.

Visualization
-------------
Scripts in `graph/` can be used for visualization and analysis. Typical steps:

1. Set `best_loss` to point to the `best_loss.pt` generated during training.
2. Provide the appropriate `--method-artifact` (e.g., `best_bcsr.pt`, `best_rel.pt`, or `bestCandidate.pt` for MIND).

**Data**
--------
The data is not uploaded yet since it required big spaces, the data use the MOT dataset format (separated by space) and store in folder 'data'. the data should consist of 3 main folder which is eval,test,train.

Appendix
--------
Additional experimental details are available in [ICRA27-Appendix.pdf](ICRA27-Appendix.pdf).

Project structure (key files)
----------------------------
- `mainContinual.py` — continual learning entry points and configs
- `models/` — trained model checkpoints organized by scene
- `graph/` — visualization and result plots
- `config/` — dataset and scene-specific configs
- `requirements.txt` — Python dependencies

Acknowledgements
----------------
This project builds on and extends concepts from SocialVAE (https://github.com/xupei0610/socialvae).

Reference
---------
Xu, J.-B., Hayet, G., and Karamouzas, I., "SocialVAE: Human trajectory prediction using timewise latents," in ECCV 2022, pp. 511–528.

