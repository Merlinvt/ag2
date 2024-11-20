import os
from typing import Dict, List, Optional, Union, Any, Callable, Literal
from dataclasses import dataclass, field
import json
import logging
import tempfile
from datetime import datetime
from .orchestrator_prompts import (
    ORCHESTRATOR_SYSTEM_MESSAGE,
    ORCHESTRATOR_CLOSED_BOOK_PROMPT,
    ORCHESTRATOR_PLAN_PROMPT,
    ORCHESTRATOR_LEDGER_PROMPT,
    ORCHESTRATOR_UPDATE_FACTS_PROMPT,
    ORCHESTRATOR_UPDATE_PLAN_PROMPT,
    ORCHESTRATOR_GET_FINAL_ANSWER,
    ORCHESTRATOR_REPLAN_PROMPT,
    ORCHESTRATOR_SYNTHESIZE_PROMPT
)
import random
from autogen.agentchat import Agent, ConversableAgent, UserProxyAgent, ChatResult
from autogen.agentchat.contrib.orchestrator_message_handler import OrchestratorMessageHandler
from autogen.agentchat.contrib.orchestrator_prompt_handler import OrchestratorPromptHandler
from autogen.agentchat.contrib.orchestrator_state import OrchestratorState

import logging
import re

logger = logging.getLogger(__name__)

class OrchestratorAgent(ConversableAgent):
    DEFAULT_SYSTEM_MESSAGES = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_MESSAGE}
    ]

    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = None,
        system_messages: List[Dict] = DEFAULT_SYSTEM_MESSAGES,
        closed_book_prompt: str = ORCHESTRATOR_CLOSED_BOOK_PROMPT,
        plan_prompt: str = ORCHESTRATOR_PLAN_PROMPT,
        synthesize_prompt: str = ORCHESTRATOR_SYNTHESIZE_PROMPT,
        ledger_prompt: str = ORCHESTRATOR_LEDGER_PROMPT,
        update_facts_prompt: str = ORCHESTRATOR_UPDATE_FACTS_PROMPT,
        update_plan_prompt: str = ORCHESTRATOR_UPDATE_PLAN_PROMPT,
        replan_prompt: str = ORCHESTRATOR_REPLAN_PROMPT,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        max_stalls_before_replan: int = 3,
        silent: Optional[bool] = None,
        agents: Optional[List[Agent]] = None,
        max_replans: int = 3,
        return_final_answer: bool = False,
        **kwargs
    ):
        super().__init__(
            name=name,
            system_message=system_messages[0]["content"],
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
            chat_messages=chat_messages,
            silent=silent,
        )

        logger.info("Initializing OrchestratorAgent...")

        # prompt-based parameters
        self._closed_book_prompt = closed_book_prompt
        self._plan_prompt = plan_prompt
        self._synthesize_prompt = synthesize_prompt
        self._ledger_prompt = ledger_prompt
        self._update_facts_prompt = update_facts_prompt
        self._update_plan_prompt = update_plan_prompt
        self._replan_prompt = replan_prompt

        # Enhanced state tracking
        self._state = {
            "task": "",
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

        # Initialize handlers
        self._state_manager = OrchestratorState()
        self._message_handler = OrchestratorMessageHandler(self._state_manager)
        self._prompt_handler = OrchestratorPromptHandler(self)
        
        # Configuration
        self._max_stalls_before_replan = max_stalls_before_replan
        self._max_replans = max_replans
        self._return_final_answer = return_final_answer

        # Team setup
        self.agents = agents or []
        self._team_description = self._prompt_handler.generate_team_description(self.agents)
        
        # Flag to identify orchestrator
        self.is_orchestrator = True

    def _handle_error_message(self, error_message: Dict) -> None:
        """Handle error messages by delegating to message handler."""
        self._message_handler.handle_error(error_message)

    def _should_attempt_recovery(self) -> bool:
        """Determine if error recovery should be attempted."""
        return (
            self._state_manager.error_recovery_attempts < 3
            and self._state_manager.replan_count < self._max_replans
        )

    def _initiate_error_recovery(self) -> None:
        """Initiate error recovery process."""
        logger.info("Initiating error recovery...")
        self._state_manager.increment_error_recovery_attempts()
        
        if self._state_manager.restore_last_successful_state():
            logger.info("Restored last successful state")
            self._state_manager.increment_replan_count()
            self._update_facts_and_plan()
        else:
            logger.warning("No successful state to restore")


    def _get_last_n_responses(self, n: int) -> List[str]:
        """Get the last n responses from the conversation."""
        responses = []
        for msg in reversed(self.chat_messages.get(self, [])):
            if msg["role"] == "assistant":
                responses.append(msg["content"])
            if len(responses) >= n:
                break
        return responses

    def _save_successful_state(self) -> None:
        """Save current state as last successful state."""
        self._state_manager.save_successful_state()

    def initiate_chat(self, message: str, sender: Optional[Agent] = None):
        """Start the orchestrated conversation with enhanced error handling."""
        logger.info(f"Starting orchestrated conversation with task: {message}")
        self._state["task"] = message
        
        try:
            # Initial analysis of the task
            self._state["facts"] = self._prompt_handler.analyze_task(message)
            logger.info(f"Task analysis complete. Facts gathered: {self._state['facts']}")
        
            # Create initial plan
            self._state["plan"] = self._prompt_handler.create_plan(self._team_description)
            logger.info(f"Initial plan created: {self._state['plan']}")
            
            # Save initial state and reset counters
            self._save_successful_state()
            self._state_manager.reset_stall_count()
            self._state_manager.update_state({"replan_count": 0})
            
            # Create initial synthesis message
            synthesized_prompt = self._prompt_handler.get_synthesize_prompt(
                self._state["task"], 
                self._team_description, 
                self._state["facts"], 
                self._state["plan"]
            )
            
            self.chat_messages.setdefault(self, []).append({"role": "user", "content": synthesized_prompt})

            for i in range(10):
            #while True: TODO: uncomment once properly tested. Maximum to prevent infinite loop
                next_speaker = self._select_next_speaker()
                if next_speaker is None:
                    break

                if next_speaker == self:
                    final_answer = self._prepare_final_answer()
                    if final_answer:

                        self.chat_messages.setdefault(self, []).append({
                            "role": "assistant",
                            "content": final_answer,
                            "name": self.name
                        })
                        return final_answer
                    break

                # For other agents, use executor
                executor = UserProxyAgent(
                    name="executor",
                    llm_config=self.llm_config,
                    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
                    max_consecutive_auto_reply=10,
                    human_input_mode="NEVER",
                    description="executor. execute the code written by the coder and report the result. When the task is completed, Return the answer and end your massage with TERMINATE.",
                    code_execution_config={
                        "work_dir": "coding",
                        "use_docker": False,
                    }
                )
                # Let the speaker generate a response
                response = executor.initiate_chat(recipient=next_speaker,message=synthesized_prompt)

                response_summary = response.summary if isinstance(response, ChatResult) else str(response)
                if self._message_handler.check_for_stall(response_summary, self._prompt_handler.update_ledger(
                    self._state["task"],
                    self._team_description,
                    [agent.name for agent in self.agents],
                    self.chat_messages
                )):
                    if self._state["stall_count"] >= self._max_stalls_before_replan:
                        self._handle_error_message({"error_type": "conversation_stall"})
                        if not self._should_attempt_recovery():
                            break
                
                # Add response to chat history
                if response_summary:
                    self.chat_messages.setdefault(self, []).append({
                        "role": "assistant",
                        "content": response_summary,
                        "name": next_speaker.name
                    })
                    
                    # Update state tracking
                    self._state["last_speaker"] = next_speaker
                    if "error" not in response_summary.lower():
                        self._save_successful_state()
                
        except Exception as e:
            logger.error(f"Error in orchestrated conversation: {str(e)}", exc_info=True)
            self._handle_error_message({
                "error_type": "system_error",
                "context": {"error": str(e)}
            })


    def _should_continue(self, ledger: Dict[str, Any]) -> bool:
        """Determine if the conversation should continue based on ledger state."""
        if ledger["is_request_satisfied"]["answer"]:
            if self._return_final_answer:
                self._prepare_final_answer()
            return False
            
        stalled = ledger["is_in_loop"]["answer"] or not ledger["is_progress_being_made"]["answer"]
        if stalled:
            self._state["stall_count"] += 1
            if self._state["stall_count"] >= self._max_stalls_before_replan:
                if self._state["replan_count"] < self._max_replans:
                    new_facts, new_plan = self._prompt_handler.update_facts_and_plan(
                        self._state,
                        self._team_description
                    )
                    self._state["facts"] = new_facts
                    self._state["plan"] = new_plan
                    self._state["replan_count"] += 1
                    self._state["stall_count"] = 0
                    
                    # Reset chat history and broadcast new plan
                    self.clear_history()
                    for agent in self.agents:
                        if isinstance(agent, ConversableAgent):
                            agent.clear_history()
                            
                    return True
                return False
        else:
            self._state["stall_count"] = 0
            
        return True

    def _prepare_final_answer(self) -> str:
        """Prepare a final answer summarizing the conversation."""
        return self.generate_reply(
            messages=[{
                "role": "system",
                "content": ORCHESTRATOR_GET_FINAL_ANSWER.format(task=self._state["task"])
            }]
        )

    def _select_next_speaker(self) -> Optional[Agent]:
        """Select the next speaker based on the orchestrator's ledger analysis."""
        try:
            # Update facts and plan if needed
            if self._state["replan_count"] > 0:
                self._update_facts_and_plan()
                self._state["replan_count"] = 0
                
            # Get ledger state
            ledger = None
            try:
                # Create the ledger prompt first
                ledger_prompt = self._ledger_prompt.format(
                    task=self._state["task"],
                    team_description=self._team_description,
                    agent_roles=json.dumps([agent.name for agent in self.agents])
                ) 
                # Start with system message
                messages = [
                ]
                
                # Add chat history if it exists
                if self.chat_messages:
                    for agent_messages in self.chat_messages.values():
                        for msg in agent_messages:
                            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                messages.append(msg)
                
                # Add ledger prompt as final assistant message
                messages.append({
                    "role": "assistant",
                    "content": ledger_prompt
                })
                
                # Get ledger state with complete message history
                response = self.generate_reply(
                    messages=messages
                )
                
                try:
                    ledger = self._prompt_handler._clean_and_parse_json(response)
                    if not self._prompt_handler._validate_ledger(ledger):
                        logger.error("Invalid ledger structure")
                        return None
                        self._format_json_str
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse ledger: {response}", exc_info=True)

                    logger.error(f"Failed to parse ledger: {response}")
                    return None
            
            except Exception as e:
                logger.warning(f"Error getting ledger: {e}. Response content: {response if 'response' in locals() else 'No response'}")
                return None

            if not ledger or not self._should_continue(ledger):
                if self._return_final_answer:
                    return self  # Let orchestrator give final answer
                return None 
        
            # Get next speaker name from ledger
            next_speaker_name = ledger["next_speaker"]["answer"]
            
            # Find the agent with matching name
            for agent in self.agents:
                if agent.name == next_speaker_name:
                    self._state["stall_count"] = 0  # Reset stall counter on success
                    return agent
                
            # If speaker not found, handle as a stall
            self._state["stall_count"] += 1
        
        except Exception as e:
            # Increment stall counter
            self._state["stall_count"] += 1
            logger.warning(f"Error in speaker selection: {e}. Stall count: {self._state['stall_count']}")
            
            # Check if we should replan
            if self._state["stall_count"] >= self._max_stalls_before_replan:
                self._state["replan_count"] += 1
                
                # Check if we've exceeded max replans
                if self._state["replan_count"] > self._max_replans:
                    logger.warning("Exceeded maximum replans, ending conversation")
                    return None       
        return None  # Default case - end conversation if no speaker found
    
