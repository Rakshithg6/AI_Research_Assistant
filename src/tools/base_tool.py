from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool's functionality."""
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.description}" 