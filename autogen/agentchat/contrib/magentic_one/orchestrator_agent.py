import json
import traceback
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from .messages import BroadcastMessage
from .orchestrator_prompts import (
    ORCHESTRATOR_SYSTEM_MESSAGE,
    ORCHESTRATOR_CLOSED_BOOK_PROMPT,
    ORCHESTRATOR_PLAN_PROMPT,
    ORCHESTRATOR_LEDGER_PROMPT,
    ORCHESTRATOR_UPDATE_FACTS_PROMPT,
    ORCHESTRATOR_UPDATE_PLAN_PROMPT,
    ORCHESTRATOR_GET_FINAL_ANSWER,
    ORCHESTRATOR_REPLAN_PROMPT,
    ORCHESTRATOR_SYNTHESIZE_PROMPT,
)


from autogen.agentchat import Agent, ConversableAgent, UserProxyAgent, ChatResult
from autogen.logger import FileLogger

logger: FileLogger = FileLogger(config={})


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
        code_execution_config: Union[Dict, Literal[False]] = {
            "work_dir": "coding",
            "use_docker": False,
        },
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
        silent: Optional[bool] = None,
        agents: Optional[List[ConversableAgent]] = [],
        max_rounds: int = 20, 
        max_stalls_before_replan: int = 3,
        max_replans: int = 3,
        return_final_answer: bool = False,
        agent_whole_history: bool = True,
        **kwargs
    ):
        super().__init__(
            name=name,
            system_message=system_messages[0]["content"],
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=False,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
            chat_messages=chat_messages,
            silent=silent,
        )

        self._system_messages = system_messages
        self._closed_book_prompt = closed_book_prompt
        self._plan_prompt = plan_prompt
        self._synthesize_prompt = synthesize_prompt
        self._ledger_prompt = ledger_prompt
        self._update_facts_prompt = update_facts_prompt
        self._update_plan_prompt = update_plan_prompt


        if chat_messages is not None:
            # Copy existing messages into defaultdict
            for agent, messages in chat_messages.items():
                for message in messages:
                    self._oai_messages[agent]._append_oai_message([message])
        self._agents = agents if agents is not None else []
            
        self._should_replan = True
        self._max_stalls_before_replan = max_stalls_before_replan
        self._stall_counter = 0
        self._max_replans = max_replans
        self._replan_counter = 0
        self._return_final_answer = return_final_answer
        self._max_rounds = max_rounds
        self._current_round = 0

        self._team_description = ""
        self._task = ""
        self._facts = ""
        self._plan = ""

    def broadcast_message(self, message: Dict[str, Any], sender: Optional[ConversableAgent] = None) -> None:
        """Broadcast a message to all agents except the sender."""
        broadcast_msg = BroadcastMessage(content=message)
        for agent in self._agents:
            if agent != sender:
                agent._append_oai_message(broadcast_msg.content, "assistant", sender or self, is_sending=False)

    def _get_plan_prompt(self, team: str) -> str:
        return self._plan_prompt.format(team_description=team)

    def _get_synthesize_prompt(self, task: str, team: str, facts: str, plan: str) -> str:
        return self._synthesize_prompt.format(
            task=task,
            team=team,
            facts=facts,
            plan=plan,
        )

    def _get_ledger_prompt(self, task: str, team: str, names: List[str]) -> str:
        return self._ledger_prompt.format(
            task=task, 
            team_description=team,  
            agent_roles=names 
        )

    def _get_update_facts_prompt(self, task: str, facts: str) -> str:
        return self._update_facts_prompt.format(
            task=task, 
            previous_facts=facts  
        )

    def _get_update_plan_prompt(self, team: str) -> str:
        return self._update_plan_prompt.format(team_description=team)  


    def _get_closed_book_prompt(self, task: str) -> str:
        return self._closed_book_prompt.format(task=task)

    def _get_team_description(self) -> str:
        """Generate a description of the team's capabilities."""
        team_description = ""
        for agent in self._agents:
            team_description += f"{agent.name}: {agent.description}\n"
        return team_description

    def _get_team_names(self) -> List[str]:
        return [agent.name for agent in self._agents]

    def _initialize_task(self, task: str) -> None:
        # called the first time a task is received
        self._task = task
        self._team_description = self._get_team_description()

        # Shallow-copy the conversation
        planning_conversation = [m for m in self._oai_messages[self]]

        # 1. GATHER FACTS
        # create a closed book task and generate a response and update the chat history
        planning_conversation.append(
            {"role": "user", "content": self._get_closed_book_prompt(self._task)}
        )
        is_valid_response, response = self.generate_oai_reply(messages=self._system_messages + planning_conversation)

        assert is_valid_response
        assert isinstance(response, str)
        self._facts = response
        planning_conversation.append(
            {"role": "assistant", "content": self._facts}
            )

        # 2. CREATE A PLAN
        ## plan based on available information
        planning_conversation.append(
            {"role": "user", "content": self._get_plan_prompt(self._team_description)}
        )

        is_valid_response, response = self.generate_oai_reply(
            messages=self._system_messages + planning_conversation
        )

        assert is_valid_response
        assert isinstance(response, str)
        self._plan = response


    def _update_facts_and_plan(self) -> None:
        # called when the orchestrator decides to replan

        planning_conversation = [m for m in self._oai_messages[self]]

        # Update the facts
        planning_conversation.append(
            {"role": "user", "content": self._get_update_facts_prompt(self._task, self._facts)}
        )

        is_valid_response, response = self.generate_oai_reply(messages=self._system_messages + planning_conversation)

        assert is_valid_response
        assert isinstance(response, str)

        self._facts = response
        planning_conversation.append(
            {"role": "assistant", "content": self._facts}
        )

        # Update the plan
        planning_conversation.append(
            {"role": "user", "content": self._get_update_plan_prompt(self._team_description)}
        )

        is_valid_response, response = self.generate_oai_reply(messages=self._system_messages + planning_conversation)

        assert is_valid_response
        assert isinstance(response, str)

        self._plan = response


    def update_ledger(self) -> Dict[str, Any]:
        # updates the ledger at each turn
        max_json_retries = 10

        team_description = self._get_team_description()
        names = self._get_team_names()
        ledger_prompt = self._get_ledger_prompt(self._task, team_description, names)

        ledger_user_messages = [
            {"role": "user", "content": ledger_prompt}
        ]
        # retries in case the LLM does not return a valid JSON
        assert max_json_retries > 0
        for _ in range(max_json_retries):
            messages = self._system_messages + self._oai_messages[self] + ledger_user_messages

            is_valid_response, ledger_str = self.generate_oai_reply(
                messages=messages,
            )

            assert is_valid_response
            assert isinstance(ledger_str, str)

            try:
                ledger_dict: Dict[str, Any] = self._clean_and_parse_json(ledger_str)

                required_keys = [
                    "is_request_satisfied",
                    "is_in_loop",
                    "is_progress_being_made", 
                    "next_speaker",
                    "instruction_or_question",
                ]
                key_error = False
                for key in required_keys:
                    if key not in ledger_dict:
                        ledger_user_messages.append(
                            {"role": "assistant", "content": ledger_str})
                        ledger_user_messages.append(
                            {"role": "user", "content": f"KeyError: '{key}'"}
                        )
                        key_error = True
                        break
                    if "answer" not in ledger_dict[key]:
                        ledger_user_messages.append(
                            {"role": "assistant", "content": ledger_str})
                        ledger_user_messages.append(
                            {"role": "user", "content": f"KeyError: '{key}.answer'"}
                        )
                        key_error = True
                        break
                if key_error:
                    continue
                return ledger_dict
            except json.JSONDecodeError as e:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={
                        "stage": "error",
                        "error_type": "JSONDecodeError", 
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                        "ledger_str": ledger_str
                    }
                )
                raise e

        raise ValueError("Failed to parse ledger information after multiple retries.")

    def _prepare_final_answer(self) -> str:
        # called when the task is complete

        final_message = {"role": "user", "content": ORCHESTRATOR_GET_FINAL_ANSWER.format(task=self._task)}
        is_valid_response, response = self.generate_oai_reply(
            messages=self._system_messages + self._oai_messages[self] + [final_message]
            )

        assert is_valid_response
        assert isinstance(response, str)
        
        return response


    def _select_next_agent(self, task: dict | str) -> Optional[ConversableAgent]:
        """Select the next agent to act based on the current state."""
        taskstr: str = ""
        if isinstance(task, dict): 

            if isinstance(task["content"], str):
                taskstr = task["content"]
            elif isinstance(task["content"], list) and task["content"][0]["type"] == "text" and isinstance(task["content"][0]["text"], str):
                taskstr = task["content"][0]["text"]
            elif isinstance(task["content"], list) and task["content"][1]["type"] == "text" and isinstance(task["content"][1]["text"], str):
                taskstr = task["content"][1]["text"]
            else:
                raise ValueError(f"Invalid task format: {task}")
        elif isinstance(task, str):
            taskstr = task

        if taskstr.strip() == "":
            return None  # Empty task

        if not self._task:
            self._initialize_task(taskstr)
            # Verify initialization
            assert len(self._task) > 0
            assert len(self._facts) > 0
            assert len(self._plan) > 0
            assert len(self._team_description) > 0

            # Create initial plan message
            synthesized_prompt = self._get_synthesize_prompt(
                self._task, self._team_description, self._facts, self._plan
            )

            # Initialize state
            self._replan_counter = 0
            self._stall_counter = 0
            
            # Log the initial plan
            logger.log_event(
                source=self.name,
                name="thought",
                data={"stage": "initial_plan", "plan": synthesized_prompt}
            )

            # Add to chat history
            self._append_oai_message(synthesized_prompt, "assistant", self, True)

            # Add initial plan to chat history only
            return self._select_next_agent(synthesized_prompt)

        # Orchestrate the next step
        ledger_dict = self.update_ledger()
        logger.log_event(
            source=self.name,
            name="thought",
            data={
                "stage": "ledger_update",
                "content": json.dumps(ledger_dict, indent=2)
            }
        )

        # Task is complete
        if ledger_dict["is_request_satisfied"]["answer"] is True:
            logger.log_event(
                source=self.name,
                name="thought",
                data={
                    "stage": "task_complete",
                    "message": "Request satisfied"
                }
            )
            
            if self._return_final_answer:
                # generate a final message to summarize the conversation
                final_answer = self._prepare_final_answer()
                logger.log_event(
                    source=self.name,
                    name="final_answer",
                    data={"answer": final_answer}
                )
                
                # Add final answer to chat history
                final_msg = {"role": "assistant", "content": final_answer}
                self._append_oai_message(final_msg, "assistant", self, True)
                

            return None

        # Stalled or stuck in a loop
        stalled = ledger_dict["is_in_loop"]["answer"] or not ledger_dict["is_progress_being_made"]["answer"]
        if stalled:
            self._stall_counter += 1

            # We exceeded our stall counter, so we need to replan, or exit
            if self._stall_counter > self._max_stalls_before_replan:
                self._replan_counter += 1
                self._stall_counter = 0

                # We exceeded our replan counter
                if self._replan_counter > self._max_replans:
                    logger.log_event(
                        source=self.name,
                        name="thought",
                        data={
                            "stage": "termination",
                            "reason": "Replan counter exceeded"
                        }
                    )
                    return None
                # Let's create a new plan
                else:
                    logger.log_event(
                        source=self.name,
                        name="thought",
                        data={
                            "stage": "replan",
                            "reason": "Stalled ... Replanning .."
                        }
                    )

                    # Update our plan.
                    self._update_facts_and_plan()

                    # Preserve initial task message
                    initial_task = self._oai_messages[self][0]
                    
                    # Reset orchestrator history
                    self._oai_messages[self] = [initial_task]
                    
                    # Reset all agents while preserving system messages
                    for agent in self._agents:
                        if agent._oai_system_message:
                            system_msg = agent._oai_system_message[0]
                            agent.reset()
                            agent._oai_system_message = [system_msg]

                    # Send everyone the NEW plan
                    synthesized_prompt = self._get_synthesize_prompt(
                        self._task, self._team_description, self._facts, self._plan
                    )
                    
                    # Broadcast new plan to all agents
                    self.broadcast_message(
                        {"role": "assistant", "content": synthesized_prompt}
                    )

                    logger.log_event(
                        source=self.name,
                        name="thought",
                        data={
                            "stage": "new_plan",
                            "plan": synthesized_prompt
                        }
                    )

                    synthesized_message = {"role": "assistant", "content": synthesized_prompt}
                    self._append_oai_message(synthesized_message, "assistant", self, True)

                    # Answer from this synthesized message
                    return self._select_next_agent(synthesized_prompt)

        # Select next agent and send instruction
        next_agent_name = ledger_dict["next_speaker"]["answer"]
        for agent in self._agents:
            if agent.name == next_agent_name:
                instruction = ledger_dict["instruction_or_question"]["answer"]
                
                # Log the instruction
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={
                        "stage": f"-> {next_agent_name}",
                        "instruction": instruction
                    }
                )
                
                # Update chat history
                instruction_msg = {"role": "assistant", "content": instruction}
                self._append_oai_message(instruction_msg, "assistant", self, True)
                
                # Broadcast instruction to all agents
                self.broadcast_message(instruction_msg)
                return agent

        return None

    async def a_generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, Dict, None]:
        """Start the orchestration process with an initial message/task."""
        # Reset state
        self._current_round = 0
        self._oai_messages.clear()
        for agent in self._agents:
            agent.reset()

        if messages is None:
            message = self.get_human_input("Please provide the task: ")
        else:
            message = messages[-1]["content"]

        # Initialize the first agent selection
        next_agent = self._select_next_agent(message)
        
        last_summary = ""
        # Continue orchestration until max rounds reached or no next agent
        while next_agent is not None and self._current_round < self._max_rounds:
            self._current_round += 1
            
            instructions = self._oai_messages[self][-1] if self._oai_messages[self] else None
            if not instructions:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={
                        "stage": "error",
                        "message": "No message found in chat history"
                    }
                )
                break
                
            response = await next_agent.a_generate_reply(messages=[instructions], sender=self)

            if isinstance(response, str):
                response = response
            elif isinstance(response, dict):
                response = response["content"]
            else:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={
                        "stage": "error",
                        "message": f"Invalid response type: {type(response)}"
                    }
                )
                break
            response_msg = {"role": "user", "content": response}

            # Broadcast response to all agents
            self.broadcast_message(response_msg, sender=next_agent)

            was_appended = self._append_oai_message(response_msg, "user", self, is_sending=False)
            if not was_appended:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={
                        "stage": "error",
                        "message": "Failed to append message to OAI messages"
                    }
                )
                break

            next_agent = self._select_next_agent(response_msg)
                    
            if self._current_round >= self._max_rounds:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={
                        "stage": "max_rounds_reached",
                        "max_rounds": self._max_rounds
                    }
                )
                
        # Track final state
        final_state = {
            "rounds_completed": self._current_round,
            "replans": self._replan_counter,
            "stalls": self._stall_counter,
            "task_completed": next_agent is None and self._current_round < self._max_rounds
        }
        logger.log_event(
            source=self.name,
            name="thought",
            data={
                "stage": "final_state",
                "state": final_state
            }
        )
        
        # Return chat result with all relevant info
        return self._oai_messages[self][-1]["content"]


    def _clean_and_parse_json(self, content: str) -> Dict[str, Any]:
        """Clean and parse JSON content from various formats."""
        if not content or not isinstance(content, str):
            raise ValueError("Content must be a non-empty string")

        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            parts = content.split("```json")
            if len(parts) > 1:
                content = parts[1].split("```")[0].strip()
        elif "```" in content:  # Handle cases where json block might not be explicitly marked
            parts = content.split("```")
            if len(parts) > 1:
                content = parts[1].strip()  # Take first code block content
        
        # Find JSON-like structure if not in code block
        if not content.strip().startswith('{'):
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                raise ValueError(
                    f"Could not find valid JSON structure in content. "
                    f"Content must contain a JSON object enclosed in curly braces. "
                    f"Received: {content}"
                )
            content = json_match.group(0)

        # Now clean for parsing
        try:
            # First try parsing the cleaned but formatted content
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try more aggressive cleaning
            cleaned_content = re.sub(r'[\n\r\t]', ' ', content)  # Replace newlines/tabs with spaces
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content)  # Normalize whitespace
            cleaned_content = re.sub(r'\\(?!["\\/bfnrt])', '', cleaned_content)  # Remove invalid escapes
            cleaned_content = re.sub(r',(\s*[}\]])', r'\1', cleaned_content)  # Remove trailing commas
            cleaned_content = re.sub(r'([{,]\s*)(\w+)(?=\s*:)', r'\1"\2"', cleaned_content)  # Quote unquoted keys
            cleaned_content = cleaned_content.replace("'", '"')  # Standardize quotes
            
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError as e:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={
                        "stage": "json_error",
                        "original_content": content,
                        "cleaned_content": cleaned_content,
                        "message": str(e)
                    }
                )
                raise ValueError(f"Failed to parse JSON after cleaning. Error: {str(e)}")
