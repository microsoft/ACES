"""
SABER-to-Inspect-AI Task Adapter

This module provides adapters to convert SABER tasks and workflows into
Inspect-AI's TaskProfile format for seamless textual UI integration.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import TaskInfo


class MockEvalConfig:
    """Mock EvalConfig that satisfies Inspect-AI's expectations."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.score_display = config_dict.get("score_display", True)
        self.log_level = config_dict.get("log_level", "INFO")
        self.fail_on_error = config_dict.get("fail_on_error", False)
        self.epochs = config_dict.get("epochs", 1)

        # Set all config values as attributes
        for key, value in config_dict.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def model_dump(self, exclude_none: bool = False) -> Dict[str, Any]:
        """Mock model_dump method expected by Inspect-AI."""
        result = {}
        for key, value in self.__dict__.items():
            if not exclude_none or value is not None:
                result[key] = value
        return result


class MockGenerateConfig:
    """Mock GenerateConfig that satisfies Inspect-AI's expectations."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.max_retries = config_dict.get("max_retries", 3)
        self.timeout = config_dict.get("timeout", 60)
        self.max_connections = config_dict.get("max_connections", 10)

        # Set all config values as attributes
        for key, value in config_dict.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def model_dump(self, exclude_none: bool = False) -> Dict[str, Any]:
        """Mock model_dump method expected by Inspect-AI."""
        result = {}
        for key, value in self.__dict__.items():
            if not exclude_none or value is not None:
                result[key] = value
        return result


class MockEvalStats:
    """Mock EvalStats that satisfies Inspect-AI's expectations."""

    def __init__(self) -> None:
        self.model_usage: Dict[str, Any] = {}  # Dictionary for model usage (not list!)
        self.total_time = 30.0  # Mock timing
        self.total_samples = 1
        self.completed_samples = 1

        # Add datetime fields that Inspect-AI expects
        current_time = datetime.now()
        self.started_at = current_time.isoformat()
        self.completed_at = (current_time).isoformat()

        # Additional fields that might be expected
        self.eval_time = self.total_time
        self.token_usage = 0

    def model_dump(self, exclude_none: bool = False) -> Dict[str, Any]:
        """Mock model_dump method expected by Inspect-AI."""
        return {
            "model_usage": self.model_usage,
            "total_time": self.total_time,
            "total_samples": self.total_samples,
            "completed_samples": self.completed_samples,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "eval_time": self.eval_time,
            "token_usage": self.token_usage,
        }


class MockScore:
    """Mock Score object that satisfies Inspect-AI's expectations."""

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
        self.metrics: Dict[str, Any] = {}  # Empty dict for metrics

    def model_dump(self, exclude_none: bool = False) -> Dict[str, Any]:
        """Mock model_dump method expected by Inspect-AI."""
        return {"name": self.name, "value": self.value, "metrics": self.metrics}


class MockEvalResults:
    """Mock EvalResults that satisfies Inspect-AI's expectations."""

    def __init__(self) -> None:
        self.scores = [MockScore("security_score", 0.85)]  # Mock scores as objects
        self.metadata = {"saber_evaluation": True}

    def model_dump(self, exclude_none: bool = False) -> Dict[str, Any]:
        """Mock model_dump method expected by Inspect-AI."""
        return {"scores": [score.model_dump(exclude_none) for score in self.scores], "metadata": self.metadata}


@dataclass
class SABERTaskProfile:
    """
    Adapter class that converts SABER task information into Inspect-AI TaskProfile format.
    """

    name: str
    file: Optional[str]
    model: str
    dataset: str
    scorer: str
    samples: int
    steps: int
    eval_config: MockEvalConfig
    task_args: Dict[str, Any]
    generate_config: MockGenerateConfig
    tags: Optional[List[str]]
    log_location: str

    # SABER-specific extensions
    saber_task_id: str
    saber_episode_id: Optional[str] = None
    saber_session_id: Optional[str] = None
    security_context: Optional[Dict[str, Any]] = None

    @classmethod
    def from_saber_task(
        cls, task_info: TaskInfo, session_context: Optional[Dict[str, Any]] = None
    ) -> "SABERTaskProfile":
        """
        Convert a SABER TaskInfo into a SABERTaskProfile suitable for Inspect-AI.

        Args:
            task_info: SABER task information
            session_context: Additional session context (episode_id, security settings, etc.)

        Returns:
            SABERTaskProfile instance ready for Inspect-AI consumption
        """
        session_context = session_context or {}

        # Extract SABER-specific metadata
        metadata = task_info.metadata or {}

        # Create proper config objects
        eval_config_dict = {
            "score_display": True,
            "log_level": "INFO",
            "fail_on_error": False,
            "epochs": 1,
            **metadata.get("eval_config", {}),
        }

        generate_config_dict = {
            "max_retries": 3,
            "timeout": 60,
            "max_connections": 10,
            **metadata.get("generate_config", {}),
        }

        return cls(
            name=task_info.name,
            file=metadata.get("source_file"),
            model=metadata.get("model_name", "saber-agent"),
            dataset=metadata.get("dataset", "saber-security-tasks"),
            scorer=metadata.get("scorer", "saber-security-scorer"),
            samples=metadata.get("sample_count", 1),
            steps=metadata.get("estimated_steps", 10),
            eval_config=MockEvalConfig(eval_config_dict),
            task_args=metadata.get("task_args", {}),
            generate_config=MockGenerateConfig(generate_config_dict),
            tags=metadata.get("tags", ["saber", "security"]),
            log_location=metadata.get("log_location", f"./logs/{task_info.task_id}.log"),
            saber_task_id=task_info.task_id,
            saber_episode_id=session_context.get("episode_id"),
            saber_session_id=session_context.get("session_id"),
            security_context=metadata.get("security_context"),
        )

    def to_inspect_ai_spec(self) -> Dict[str, Any]:
        """
        Convert to Inspect-AI TaskSpec format.

        Returns:
            Dictionary compatible with Inspect-AI's TaskSpec requirements
        """
        return {"name": self.name, "model": self.model}

    def to_inspect_ai_profile(self) -> "SABERTaskProfile":
        """
        Return self as it's already in the correct format.

        Returns:
            Self, as SABERTaskProfile is compatible with Inspect-AI's TaskProfile
        """
        return self


class SABERProgressMapper:
    """
    Maps SABER progress updates to Inspect-AI progress system.
    """

    def __init__(self, task_profile: SABERTaskProfile):
        self.task_profile = task_profile
        self.current_step = 0
        self.current_sample = 0

    def update_from_saber_progress(self, progress: float, step_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert SABER progress (0.0-1.0) to Inspect-AI progress format.

        Args:
            progress: SABER progress as float between 0.0 and 1.0
            step_info: Optional additional step information

        Returns:
            Progress update compatible with Inspect-AI
        """
        step_info = step_info or {}

        # Calculate step and sample progress
        total_steps = self.task_profile.steps
        total_samples = self.task_profile.samples

        # Update current position based on progress
        self.current_step = int(progress * total_steps)
        self.current_sample = min(int(progress * total_samples), total_samples - 1)

        return {
            "steps_completed": self.current_step,
            "steps_total": total_steps,
            "samples_completed": self.current_sample + (1 if progress == 1.0 else 0),
            "samples_total": total_samples,
            "progress_percent": int(progress * 100),
            "current_step_name": step_info.get("step_name", f"Step {self.current_step + 1}"),
            "step_details": step_info.get("details", ""),
        }


class SABERSessionContext:
    """
    Manages session context for SABER textual mode operations.
    """

    def __init__(self, session_id: str, episode_id: Optional[str] = None):
        self.session_id = session_id
        self.episode_id = episode_id
        self.tasks: List[SABERTaskProfile] = []
        self.start_time = datetime.now()
        self.metadata: Dict[str, Any] = {}

    def add_task(self, task_info: TaskInfo) -> SABERTaskProfile:
        """
        Add a SABER task to the session context.

        Args:
            task_info: SABER task information

        Returns:
            SABERTaskProfile for the added task
        """
        session_context = {"session_id": self.session_id, "episode_id": self.episode_id}

        task_profile = SABERTaskProfile.from_saber_task(task_info, session_context)
        self.tasks.append(task_profile)

        return task_profile

    def get_inspect_ai_task_specs(self) -> List[Dict[str, Any]]:
        """
        Get all tasks as Inspect-AI TaskSpec format.

        Returns:
            List of TaskSpec dictionaries
        """
        return [task.to_inspect_ai_spec() for task in self.tasks]

    def get_session_metadata(self) -> Dict[str, Any]:
        """
        Get session metadata for logging and display.

        Returns:
            Session metadata dictionary
        """
        return {
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "start_time": self.start_time.isoformat(),
            "task_count": len(self.tasks),
            "task_names": [task.name for task in self.tasks],
            **self.metadata,
        }
