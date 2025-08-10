# AirHockey RL — TD3 & SAC

![TD3 self-play](docs/td3_demo.gif)  
*TD3 self-play (TD3 agent vs TD3 agent).*

Train two reinforcement learning agents—**TD3** and **SAC**—to master a continuous-control **AirHockey** environment.  
This repo contains training scripts, demo notebooks, and experiment results.

---

## 📌 What this is
- Reinforcement learning environment for Airhockey
- Training scripts for TD3 and SAC magents  
- Quick demo notebook to render the environment  
- Presentation with experimentation results (see **docs/**)  

---

## 🎯 What it does
- Trains **TD3** and **SAC** agents on the same AirHockey game  
- Lets you easily tweak hyperparameters (e.g., policy noise, policy delay)  
- (Optional) Logs training runs to **Weights & Biases** for tracking  

---

## 🚀 How to use

**Requirements:** Python **3.10+**

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Train TD3:**

```bash
python TD3/main.py
```

**Train SAC:**

```bash
python/SAC/main.py
```

**Environment demo:** Open **example_run.ipynb** and run the cell to render a short random-action rollout.

---

## 📂 Details
Presentation of our experiments can be found at: **docs/rl-hockey-presentation.pptx**

---

## 📜 License
MIT License
