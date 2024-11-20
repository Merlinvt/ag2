from typing import Dict, List, Optional, Any
import logging
from autogen.agentchat.contrib.orchestrator_state import OrchestratorState
from autogen.agentchat import Agent

logger = logging.getLogger(__name__)

class OrchestratorMessageHandler:
    """
    Handles different types of messages for the OrchestratorAgent.
    
    This class processes various message types including broadcasts,
    resets, errors, and progress updates. It works closely with
    OrchestratorState to maintain state consistency.
    """
    
    def __init__(self, state_manager: OrchestratorState):
        """
        Initialize MessageHandler with a state manager.
        
        Args:
            state_manager (OrchestratorState): The state management instance
        """
        self._state_manager = state_manager

    def handle_broadcast(self, message: Dict, agents: List[Agent]) -> None:
        """
        Broadcast a message to all agents in the team.
        
        Args:
            message (Dict): The message to broadcast
            agents (List[Agent]): List of agents to receive the broadcast
        """
        try:
            broadcast_content = message.get("content", "")
            if not broadcast_content:
                logger.warning("Empty broadcast message received")
                return

            logger.info(f"Broadcasting message to {len(agents)} agents")
            for agent in agents:
                if hasattr(agent, 'receive_broadcast'):
                    agent.receive_broadcast(broadcast_content)
                else:
                    logger.warning(f"Agent {agent.name} does not support broadcasts")

        except Exception as e:
            logger.error(f"Error in broadcast handling: {str(e)}", exc_info=True)
            self._state_manager.add_failed_attempt({
                "error_type": "broadcast_error",
                "context": {"error": str(e)}
            })

    def handle_reset(self, preserve_task: bool = True) -> None:
        """
        Reset the orchestrator's state.
        
        Args:
            preserve_task (bool): Whether to preserve the original task
        """
        try:
            logger.info("Resetting orchestrator state")
            self._state_manager.reset(preserve_task)
            
        except Exception as e:
            logger.error(f"Error in reset handling: {str(e)}", exc_info=True)
            self._state_manager.add_failed_attempt({
                "error_type": "reset_error",
                "context": {"error": str(e)}
            })

    def handle_error(self, error_message: Dict) -> None:
        """
        Handle error messages and update state accordingly.
        
        Args:
            error_message (Dict): The error message containing error details
        """
        try:
            error_type = error_message.get("error_type", "unknown_error")
            context = error_message.get("context", {})
            
            logger.error(f"Handling error: {error_type}")
            logger.debug(f"Error context: {context}")
            
            self._state_manager.add_failed_attempt({
                "error_type": error_type,
                "context": context,
            })
            
            # Increment stall count for certain error types
            if error_type in ["conversation_stall", "response_timeout"]:
                self._state_manager.increment_stall_count()
                
        except Exception as e:
            logger.error(f"Error in error handling: {str(e)}", exc_info=True)
            self._state_manager.add_failed_attempt({
                "error_type": "error_handler_error",
                "context": {"error": str(e)}
            })

    def handle_progress(self, progress_message: Dict) -> None:
        """
        Handle progress updates and maintain progress markers.
        
        Args:
            progress_message (Dict): The progress update message
        """
        try:
            # Extract progress information
            marker = progress_message.get("marker")
            completed_subtask = progress_message.get("completed_subtask")
            
            if marker or completed_subtask:
                logger.info(f"Recording progress - Marker: {marker}, Completed subtask: {completed_subtask}")
                self._state_manager.add_progress({
                    "marker": marker,
                    "completed_subtask": completed_subtask
                })
                self._state_manager.reset_stall_count()
                
            else:
                logger.warning("Progress message received without marker or completed subtask")
                
        except Exception as e:
            logger.error(f"Error in progress handling: {str(e)}", exc_info=True)
            self._state_manager.add_failed_attempt({
                "error_type": "progress_error",
                "context": {"error": str(e)}
            })

    def check_for_stall(self, response: str, ledger: Optional[Dict] = None) -> bool:
        """
        Check if the conversation has stalled based on response and ledger state.
        
        Args:
            response (str): The response to check
            ledger (Optional[Dict]): Optional ledger state to check
            
        Returns:
            bool: True if conversation has stalled, False otherwise
        """
        # Empty or whitespace response is considered a stall
        if not response or response.strip() == "":
            self._state_manager.increment_stall_count()
            return True

        if ledger:
            # Check if progress is being made according to the ledger
            is_progress = ledger.get("is_progress_being_made", {}).get("answer", False)
            is_in_loop = ledger.get("is_in_loop", {}).get("answer", False)

            # Stall conditions:
            # 1. No progress being made
            # 2. Conversation is in a loop
            if not is_progress or is_in_loop:
                self._state_manager.increment_stall_count()
                return True

            # Reset stall count if progress is being made
            self._state_manager.reset_stall_count()
            return False

        # No ledger provided is considered a stall
        self._state_manager.increment_stall_count()
        return True
