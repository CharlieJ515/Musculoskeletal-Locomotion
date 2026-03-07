import torch.nn as nn

from .sac_mlp import SAC_MLPActor, SAC_MLPCritic
from .td3_mlp import TD3_MLPActor, TD3_MLPCritic

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "SAC_MLPActor": SAC_MLPActor,
    "SAC_MLPCritic": SAC_MLPCritic,
    "TD3_MLPActor": TD3_MLPActor,
    "TD3_MLPCritic": TD3_MLPCritic,
}
