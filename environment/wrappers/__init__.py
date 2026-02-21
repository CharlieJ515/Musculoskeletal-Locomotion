from .baby_walker import BabyWalkerWrapper, LimitForceConfig
from .composite_reward import CompositeRewardWrapper
from .frame_skip import FrameSkipWrapper
from .motion_logger import MotionLoggerWrapper
from .rescale_action import RescaleActionWrapper
from .simple_env import SimpleEnvWrapper
from .target_speed import TargetSpeedWrapper
from .target_velocity import TargetVelocityWrapper

WRAPPER_REGISTRY = {
    "MotionLoggerWrapper": MotionLoggerWrapper,
    "TargetSpeedWrapper": TargetSpeedWrapper,
    "TargetVelocityWrapper": TargetVelocityWrapper,
    "SimpleEnvWrapper": SimpleEnvWrapper,
    "CompositeRewardWrapper": CompositeRewardWrapper,
    "RescaleActionWrapper": RescaleActionWrapper,
    "FrameSkipWrapper": FrameSkipWrapper,
    "BabyWalkerWrapper": BabyWalkerWrapper,
}

__all__ = [
    "WRAPPER_REGISTRY",
    "LimitForceConfig",
]
