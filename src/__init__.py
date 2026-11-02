from .ExpertManager import ExpertManager, get_expert_manager

__all__ = ["ExpertManager", "get_expert_manager", "expert_manager"]


def __getattr__(name: str):
    if name == "expert_manager":
        return get_expert_manager()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
