from typing import Dict, Any, Optional
from datetime import datetime
import copy

class OrchestratorState:
    def __init__(self):
        self.reset()
        self._last_successful_state = None
        self._error_recovery_attempts = 0
        
    def reset(self, preserve_task: bool = True) -> None:
        """Reset state while optionally preserving the original task."""
        task = self._state["task"] if preserve_task and hasattr(self, '_state') else ""
        self._state = {
            "task": task,
            "facts": "",
            "plan": "",
            "current_phase": "",
            "last_speaker": None,
            "stall_count": 0,
            "replan_count": 0,
            "progress_markers": set(),
            "completed_subtasks": [],
            "failed_attempts": [],
        }

    def save_successful_state(self) -> None:
        """Save current state as last successful state."""
        self._last_successful_state = copy.deepcopy(self._state)

    def restore_last_successful_state(self) -> bool:
        """Restore last known good state if available."""
        if self._last_successful_state:
            self._state.update(self._last_successful_state)
            return True
        return False

    def add_failed_attempt(self, attempt: Dict) -> None:
        self._state["failed_attempts"].append(attempt)

    def add_progress(self, progress: Dict) -> None:
        if progress.get("marker"):
            self._state["progress_markers"].add(progress["marker"])
        if progress.get("completed_subtask"):
            self._state["completed_subtasks"].append(progress["completed_subtask"])

    def increment_stall_count(self) -> None:
        self._state["stall_count"] += 1

    def reset_stall_count(self) -> None:
        self._state["stall_count"] = 0

    def increment_replan_count(self) -> None:
        self._state["replan_count"] += 1

    def increment_error_recovery_attempts(self) -> None:
        self._error_recovery_attempts += 1

    def get_state(self) -> Dict:
        return self._state

    def update_state(self, updates: Dict) -> None:
        self._state.update(updates)

    @property
    def stall_count(self) -> int:
        return self._state["stall_count"]

    @property
    def replan_count(self) -> int:
        return self._state["replan_count"]

    @property
    def error_recovery_attempts(self) -> int:
        return self._error_recovery_attempts
