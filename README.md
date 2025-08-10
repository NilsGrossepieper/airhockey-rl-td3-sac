# AirHockey RL — TD3 & SAC

Train two reinforcement-learning agents—**TD3** and **SAC**—to play a continuous-control **AirHockey** environment.  
The repo includes a simple environment demo, training scripts for both agents, and slides with short clips and plots.

## What it does
- Trains TD3 and SAC agents on the same AirHockey task
- Compares behaviors with simple tweaks (e.g., policy noise, policy delay)
- Optionally logs runs to Weights & Biases

## Quick start
Requirements: Python 3.10+.

```bash
pip install -r requirements.txt
```

### Train TD3
```bash
python TD3/main.py
```

### Train SAC
```bash
python SAC/main.py
```

> Tip: add `--help` to either command to see available options.

## Environment demo
Open **`example_run.ipynb`** and run the cells to see the environment stepping with random actions and rendering.

@"
## Results & media
- Slides with short gameplay clips and plots: **docs/rl-hockey-presentation.pptx**
- TD3 self-play:

<video src="docs/td3_demo.mp4" controls muted loop playsinline width="640"></video>
"@ | Add-Content -Encoding UTF8 README.md
  ```

## Repository structure
```
TD3/                    # TD3 implementation and training entry (main.py)
SAC/                    # SAC implementation and training entry (main.py)
hockey_env/             # environment helpers/wrappers
dynamic_env.py          # environment orchestration, evaluation, logging hooks
example_run.ipynb       # quick demo notebook
docs/                   # slides, plots, gifs
requirements.txt
LICENSE
```

## License
MIT

