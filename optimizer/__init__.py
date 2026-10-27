__all__ = ["TCG"]


def __getattr__(name):
    if name == "TCG":
        from .tcg import TCG

        return TCG
    raise AttributeError(name)
