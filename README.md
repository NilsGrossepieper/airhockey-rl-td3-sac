# AirHockey RL  TD3 & SAC

This repo shows two reinforcement-learning agents for AirHockey: **TD3** and **SAC**.
We compare them and test the effect of **policy noise** and **policy delay**.

## How to run
- TD3:  python TD3/train.py --env HockeyEnv-v0 --steps 500000 --seed 0 --policy-noise 0.2 --policy-delay 2
- SAC:  python SAC/train.py  --env HockeyEnv-v0 --steps 500000 --seed 0

## What to look at
- Slides (with short clips): docs/rl-hockey-presentation.pptx

## Repo layout
TD3/  SAC/  hockey_env/  docs/  requirements.txt

## License
MIT
