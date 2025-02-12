import torch
import numpy as np

def save_model(actor, critic1, critic2, actor_target, critic1_target, critic2_target, filename="td3_model.pth"):
    """
    Save model parameters.

    Parameters:
    - actor (nn.Module): Trained actor network
    - critic1 (nn.Module): First trained critic network
    - critic2 (nn.Module): Second trained critic network
    -actor_target (nn.Module): Trained actor target network
    - critic1_target (nn.Module): First trained critic target network
    - critic2_target (nn.Module): Second trained critic target
    - filename (str): File name for saving the model
    """
    torch.save({
    "actor": actor.state_dict(),
    "critic1": critic1.state_dict(),
    "critic2": critic2.state_dict(),
    "actor_target": actor_target.state_dict(),
    "critic1_target": critic1_target.state_dict(),
    "critic2_target": critic2_target.state_dict(),
    }, filename)
    print(f"Model saved as {filename}")

def load_model(actor, critic1, critic2, actor_target, critic1_target, critic2_target, filename="td3_model.pth"):

    """
    Load model parameters from a file.

    Parameters:
    - actor (nn.Module): Actor network to load parameters into
    - critic1 (nn.Module): First critic network to load parameters into
    - critic2 (nn.Module): Second critic network to load parameters into
    -actor_target (nn.Module): Actor target network to load parameters into
    - critic1_target (nn.Module): First critic target network to load parameters into
    - critic2_target (nn.Module): Second critic target network to load parameters into
    - filename (str): File name to load the model from
    """
    checkpoint = torch.load(filename)
    actor.load_state_dict(checkpoint["actor"])
    critic1.load_state_dict(checkpoint["critic1"])
    critic2.load_state_dict(checkpoint["critic2"])
    actor_target.load_state_dict(checkpoint["actor_target"])
    critic1_target.load_state_dict(checkpoint["critic1_target"])
    critic2_target.load_state_dict(checkpoint["critic2_target"])
    print(f"Model loaded from {filename}")