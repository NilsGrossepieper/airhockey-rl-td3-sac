import torch
import numpy as np

# Define device for GPU/CPU compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(actor, critic, actor_target, critic_target, filename="td3_model.pth"):
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_target": actor_target.state_dict(),
        "critic_target": critic_target.state_dict(),
    }, filename)
    print(f"Model saved as {filename}")

def load_model(actor, critic, actor_target, critic_target, filename="td3_model.pth"):
    checkpoint = torch.load(filename, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    critic.load_state_dict(checkpoint["critic"])
    actor_target.load_state_dict(checkpoint["actor_target"])
    critic_target.load_state_dict(checkpoint["critic_target"])
    print(f"Model loaded from {filename}")