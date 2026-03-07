from .base import CompositeReward, RewardComponent
from .effort import EnergyReward, SmoothnessReward
from .forces import BodySupportReward, FootImpactPenalty
from .posture import HeadStabilityReward, UprightReward
from .task import AliveReward, FootstepReward, VelocityReward

REWARD_REGISTRY: dict[str, type[RewardComponent]] = {
    "AliveReward": AliveReward,
    "VelocityReward": VelocityReward,
    "FootstepReward": FootstepReward,
    "EnergyReward": EnergyReward,
    "SmoothnessReward": SmoothnessReward,
    "HeadStabilityReward": HeadStabilityReward,
    "UprightReward": UprightReward,
    "FootImpactPenalty": FootImpactPenalty,
    "BodySupportReward": BodySupportReward,
}
