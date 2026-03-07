import functools
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

def require_reset(fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]

        if not hasattr(self, "_was_reset"):
            raise AttributeError(
                "The object must define a boolean attribute '_was_reset' before using "
                f"{fn.__name__}()."
            )

        if not self._was_reset: # pyright: ignore[reportAttributeAccessIssue]
            raise RuntimeError("Call reset() before using this method.")

        return fn(*args, **kwargs)
    return wrapper
