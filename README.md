
---

### Or, do it with one PowerShell block (overwrites README and pushes)

```powershell
cd C:\Code\rl-hockey-homework

@"
# AirHockey RL — TD3 & SAC

![TD3 self-play](docs/td3_demo.gif)

*TD3 self-play (TD3 agent vs TD3 agent).*

Train two reinforcement-learning agents—**TD3** and **SAC**—to play a continuous-control **AirHockey** environment.  
The repo includes a simple environment demo, training scripts for both agents, and slides with short clips and plots.

## What it does
- Trains TD3 and SAC agents on the same AirHockey task
- Compares behaviors with simple tweaks (e.g., policy noise, policy delay)
- Optionally logs runs to Weights & Biases

## Quick start
Requirements: Python 3.10+.

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Train TD3
\`\`\`bash
python TD3/main.py
\`\`\`

### Train SAC
\`\`\`bash
python SAC/main.py
\`\`\`

> Tip: add \`--help\` to either command to see available options.

## Environment demo
Open **\`example_run.ipynb\`** and run the cells to see the environment stepping with random actions and rendering.

## Results & media
- Slides with short gameplay clips and plots: **\`docs/rl-hockey-presentation.pptx\`**
- Demo video file (MP4): **\`docs/td3_demo.mp4\`** (source for the GIF)

## Repository structure
\`\`\`
TD3/                    # TD3 implementation and training entry (main.py)
SAC/                    # SAC implementation and training entry (main.py)
hockey_env/             # environment helpers/wrappers
dynamic_env.py          # environment orchestration, evaluation, logging hooks
example_run.ipynb       # quick demo notebook
docs/                   # slides, plots, demo video (td3_demo.mp4) and GIF (td3_demo.gif)
requirements.txt
LICENSE
\`\`\`

## License
MIT
"@ | Set-Content -Encoding UTF8 README.md
