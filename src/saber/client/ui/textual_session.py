"""
SABER Textual Session Implementation

This module provides a dedicated textual session wrapper that runs SABER
evaluations within Inspect-AI's textual UI context, following their TaskScreenApp pattern.
"""

import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

# Inspect-AI display imports (conditional to avoid hard dependency)
try:
    from inspect_ai._display.core.active import display as get_active_display
    from inspect_ai._display.core.display import Display, TaskDisplay, TaskError, TaskScreen, TaskSpec, TaskSuccess

    INSPECT_AI_AVAILABLE = True
except ImportError:
    INSPECT_AI_AVAILABLE = False

from .models import MessageType, TaskInfo
from .textual_adapter import MockEvalResults, MockEvalStats, SABERProgressMapper, SABERSessionContext, SABERTaskProfile

logger = logging.getLogger(__name__)


class SABERTextualSession:
    """
    Dedicated textual session wrapper that runs SABER evaluations within
    Inspect-AI's textual UI context.

    This class follows the recommended Option A approach, leveraging Inspect-AI's
    TaskScreenApp infrastructure while maintaining clean separation from SABER's
    core evaluation logic.
    """

    def __init__(self, session_id: str, episode_id: Optional[str] = None):
        """
        Initialize the SABER textual session.

        Args:
            session_id: Unique identifier for this session
            episode_id: Optional episode identifier for grouping
        """
        if not INSPECT_AI_AVAILABLE:
            raise RuntimeError(
                "Inspect-AI is not available. Cannot create textual session. "
                "Please install inspect-ai or use a different UI mode."
            )

        self.session_context = SABERSessionContext(session_id, episode_id)
        self.progress_mappers: Dict[str, SABERProgressMapper] = {}
        self.task_displays: Dict[str, TaskDisplay] = {}
        self.active_tasks: Dict[str, SABERTaskProfile] = {}
        self._display: Optional[Display] = None
        self._task_screen: Optional[TaskScreen] = None

    def run_saber_evaluation(self, evaluation_func: Callable[["SABERTextualSession"], Coroutine[Any, Any, Any]]) -> Any:
        """
        Run a SABER evaluation within the textual UI context.

        This is the main entry point that follows Inspect-AI's run_task_app pattern.
        Note: This method is synchronous and handles the async/sync boundary properly.

        Args:
            evaluation_func: Async function that performs the SABER evaluation

        Returns:
            Result of the evaluation function
        """

        async def main_evaluation() -> Any:
            """Main evaluation function to be run within textual context."""
            try:
                # Get the active display (will be set by Inspect-AI's textual app)
                self._display = get_active_display()

                # Log session start
                self._display.print(f"🚀 Starting SABER textual session: {self.session_context.session_id}")

                # Run the SABER evaluation
                result = await evaluation_func(self)

                # Log session completion
                self._display.print("✅ SABER session completed successfully")

                return result

            except Exception as e:
                logger.exception("Error in SABER textual evaluation")
                if self._display:
                    self._display.print(f"💥 SABER session error: {e}")
                raise

        # Use Inspect-AI's textual display to run the evaluation
        # Note: This must be called from sync context, not async
        display = get_active_display()
        return display.run_task_app(main_evaluation)

    async def start_task_session(self, tasks: List[TaskInfo]) -> None:
        """
        Start a task session with the given tasks.

        Args:
            tasks: List of SABER tasks to execute
        """
        # Convert SABER tasks to Inspect-AI format
        task_profiles = []
        for task_info in tasks:
            profile = self.session_context.add_task(task_info)
            task_profiles.append(profile)
            self.active_tasks[task_info.task_id] = profile

            # Create progress mapper
            self.progress_mappers[task_info.task_id] = SABERProgressMapper(profile)

        # Get task specs for Inspect-AI
        task_specs = self.session_context.get_inspect_ai_task_specs()

        # Convert to Inspect-AI TaskSpec format
        inspect_task_specs = []
        for spec in task_specs:
            inspect_task_specs.append(TaskSpec(name=spec["name"], model=spec["model"]))

        if self._display:
            # Start task screen with Inspect-AI
            async with self._display.task_screen(inspect_task_specs, parallel=len(tasks) > 1) as task_screen:
                self._task_screen = task_screen

                # Initialize task displays
                for task_id, profile in self.active_tasks.items():
                    inspect_profile = self._create_inspect_task_profile(profile)
                    with self._display.task(inspect_profile) as task_display:
                        self.task_displays[task_id] = task_display

    def _create_inspect_task_profile(self, saber_profile: SABERTaskProfile) -> "SABERTaskProfile":
        """
        Return the SABER task profile as it's already compatible with Inspect-AI.

        Args:
            saber_profile: SABER task profile

        Returns:
            The same profile, as it's already in Inspect-AI compatible format
        """
        return saber_profile.to_inspect_ai_profile()

    def update_task_progress(self, task_id: str, progress: float, step_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Update progress for a specific task.

        Args:
            task_id: SABER task identifier
            progress: Progress as float between 0.0 and 1.0
            step_info: Optional step information
        """
        if task_id not in self.progress_mappers:
            logger.warning(f"Progress update for unknown task: {task_id}")
            return

        # Map SABER progress to Inspect-AI format
        mapper = self.progress_mappers[task_id]
        progress_data = mapper.update_from_saber_progress(progress, step_info)

        # Update task display if available
        if task_id in self.task_displays:
            task_display = self.task_displays[task_id]

            # Use Inspect-AI's progress context
            with task_display.progress() as progress_bar:
                progress_bar.update(progress_data["steps_completed"])

            # Update sample completion
            task_display.sample_complete(progress_data["samples_completed"], progress_data["samples_total"])

    def complete_task(self, task_id: str, success: bool = True, error: Optional[str] = None) -> None:
        """
        Mark a task as completed.

        Args:
            task_id: SABER task identifier
            success: Whether the task completed successfully
            error: Optional error message if task failed
        """
        if task_id not in self.task_displays:
            logger.warning(f"Completion for unknown task: {task_id}")
            return

        task_display = self.task_displays[task_id]

        if success:
            # Create mock stats with required attributes
            mock_stats = MockEvalStats()

            # Create mock results with required attributes
            mock_results = MockEvalResults()

            # Create success result
            result = TaskSuccess(
                samples_completed=self.active_tasks[task_id].samples, stats=mock_stats, results=mock_results
            )
        else:
            # Create error result
            result = TaskError(
                samples_completed=0,
                exc_type=RuntimeError,
                exc_value=RuntimeError(error or "Task failed"),
                traceback=None,
            )

        task_display.complete(result)

    def display_message(self, message: str, message_type: MessageType = MessageType.INFO) -> None:
        """
        Display a message in the textual UI.

        Args:
            message: Message content
            message_type: Type of message for formatting
        """
        if self._display:
            # Format message with appropriate emoji and styling
            prefix_map = {
                MessageType.INFO: "ℹ️",
                MessageType.WARNING: "⚠️",
                MessageType.ERROR: "❌",
                MessageType.SUCCESS: "✅",
                MessageType.DEBUG: "🔍",
            }

            prefix = prefix_map.get(message_type, "")
            formatted_message = f"{prefix} {message}" if prefix else message

            self._display.print(formatted_message)

    def get_session_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current session.

        Returns:
            Session metadata dictionary
        """
        return self.session_context.get_session_metadata()


def create_textual_session(session_id: str, episode_id: Optional[str] = None) -> SABERTextualSession:
    """
    Factory function to create a SABER textual session.

    Args:
        session_id: Unique session identifier
        episode_id: Optional episode identifier

    Returns:
        SABERTextualSession instance

    Raises:
        RuntimeError: If Inspect-AI is not available
    """
    return SABERTextualSession(session_id, episode_id)
