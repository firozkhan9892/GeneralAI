"""Abstract interfaces — importable shortcuts."""

from app.core.interfaces.base import IModule
from app.core.interfaces.ibrain import IBrain
from app.core.interfaces.imemory import IMemory
from app.core.interfaces.itool import ITool
from app.core.interfaces.iagent import IAgent
from app.core.interfaces.iplanner import IPlanner
from app.core.interfaces.iplugin import IPlugin
from app.core.interfaces.iworkflow import IWorkflow
from app.core.interfaces.ievent import Event, EventHandler, IEventBus

__all__ = [
    "IModule",
    "IBrain",
    "IMemory",
    "ITool",
    "IAgent",
    "IPlanner",
    "IPlugin",
    "IWorkflow",
    "Event",
    "EventHandler",
    "IEventBus",
]
