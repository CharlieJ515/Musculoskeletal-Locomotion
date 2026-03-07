import gymnasium as gym


def has_wrapper(env: gym.Env, wrapper_class: type[gym.Wrapper]) -> bool:
    wrapped = env
    while isinstance(wrapped, gym.Wrapper):
        if isinstance(wrapped, wrapper_class):
            return True
        wrapped = wrapped.env
    return False
