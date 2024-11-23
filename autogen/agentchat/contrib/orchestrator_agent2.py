import json
import logging
import traceback
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Union

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

from autogen.agentchat import Agent, ConversableAgent, UserProxyAgent, ChatResult

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
        silent: Optional[bool] = None,
        agents: Optional[List[ConversableAgent]] = [],
        max_rounds: int = 20, 
        max_stalls_before_replan: int = 3,
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

        self._system_messages = system_messages
        self._closed_book_prompt = closed_book_prompt
        self._plan_prompt = plan_prompt
        self._synthesize_prompt = synthesize_prompt
        self._ledger_prompt = ledger_prompt
        self._update_facts_prompt = update_facts_prompt
        self._update_plan_prompt = update_plan_prompt

        # Initialize chat messages dictionary with self as a key if not provided
        self._oai_messages = defaultdict(list)
        if chat_messages is not None:
            # Copy existing messages into defaultdict
            for agent, messages in chat_messages.items():
                self._oai_messages[agent] = messages.copy()
        # Ensure self has an entry
        if self not in self._oai_messages:
            self._oai_messages[self] = []
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

    def _get_closed_book_prompt(self, task: str) -> str:
        return self._closed_book_prompt.format(task=task)

    def _get_plan_prompt(self, team: str) -> str:
        return self._plan_prompt.format(team=team)

    def _get_synthesize_prompt(self, task: str, team: str, facts: str, plan: str) -> str:
        return self._synthesize_prompt.format(task=task, team=team, facts=facts, plan=plan)

    def _get_ledger_prompt(self, task: str, team: str, names: List[str]) -> str:
        return self._ledger_prompt.format(task=task, team=team, names=names)

    def _get_update_facts_prompt(self, task: str, facts: str) -> str:
        return self._update_facts_prompt.format(task=task, facts=facts)

    def _get_update_plan_prompt(self, team: str) -> str:
        return self._update_plan_prompt.format(team=team)

    def _get_team_description(self) -> str:
        """Generate a description of the team's capabilities."""
        team_description = ""
        for agent in self._agents:
            team_description += f"{agent.name}: {agent.description}\n"
        return team_description

    def _get_team_names(self) -> List[str]:
        return [agent.name for agent in self._agents]

    async def _initialize_task(self, task: str) -> None:
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
        response = await self.a_generate_reply(
            messages=self._system_messages + planning_conversation
        )

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
        response = await self.a_generate_reply(
            messages=self._system_messages + planning_conversation
        )

        assert isinstance(response, str)
        self._plan = response


    async def _update_facts_and_plan(self) -> None:
        # called when the orchestrator decides to replan

        # Shallow-copy the conversation
        planning_conversation = [m for m in self._oai_messages[self]]

        # Update the facts
        planning_conversation.append(
            {"role": "user", "content": self._get_update_facts_prompt(self._task, self._facts)}
        )
        response = await self.a_generate_reply(
            messages=self._system_messages + planning_conversation
        )

        assert isinstance(response, str)
        self._facts = response
        planning_conversation.append(
            {"role": "assistant", "content": self._facts}
        )

        # Update the plan
        planning_conversation.append(
            {"role": "user", "content": self._get_update_plan_prompt(self._team_description)}
        )
        response = await self.a_generate_reply(
            messages=self._system_messages + planning_conversation
        )

        assert isinstance(response, str)
        self._plan = response


    async def update_ledger(self) -> Dict[str, Any]:
        # updates the ledger at each turn
        max_json_retries = 10

        team_description = self._get_team_description()
        names = self._get_team_names()
        ledger_prompt = self._get_ledger_prompt(self._task, team_description, names)

        ledger_user_messages= [
            {"role": "user", "content": ledger_prompt}
        ]
        # retries in case the LLM does not return a valid JSON
        assert max_json_retries > 0
        for _ in range(max_json_retries):
            ledger_str = await self.a_generate_reply(
                messages=self._system_messages + self._oai_messages[self] + ledger_user_messages,
                json_output=True,
            )
            # TODO: json marchall

            try:
                assert isinstance(ledger_str, str)
                ledger_dict: Dict[str, Any] = json.loads(ledger_str)
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

                logger.error("An error occurred in update_ledger: %s", traceback.format_exc())
                
                logger.error(
                    "Failed to parse ledger information: %s", ledger_str
                )
                raise e

        raise ValueError("Failed to parse ledger information after multiple retries.")

    async def _prepare_final_answer(self) -> str:
        # called when the task is complete

        final_message = {"role": "user", "content": ORCHESTRATOR_GET_FINAL_ANSWER.format(task=self._task)}
        response = await self.a_generate_reply(
            messages=self._system_messages + self._oai_messages[self] + [final_message]
        )

        assert isinstance(response, str)
        return response


    async def _select_next_agent(self, message: Union[str, Dict[str, str]]) -> Optional[ConversableAgent]:
        """Select the next agent to act based on the current state."""
        task = message if isinstance(message, str) else message["content"]
        if len(self._task) == 0:
            await self._initialize_task(task)

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
            logger.info(f"Initial plan:\n{synthesized_prompt}")

            # Add to chat history
            self._append_oai_message({"role": "assistant", "content": synthesized_prompt}, "assistant", self, True)

            # Share plan with all agents
            for agent in self._agents:
                await self.a_send(synthesized_prompt, agent)

            return await self._select_next_agent(synthesized_prompt)

        # Orchestrate the next step
        ledger_dict = await self.update_ledger()
        logger.info(
                f"Updated Ledger:\n{json.dumps(ledger_dict, indent=2)}"
        )

        # Task is complete
        if ledger_dict["is_request_satisfied"]["answer"] is True:
            logger.info(
                    "Request satisfied."
            )
            if self._return_final_answer:
                # generate a final message to summarize the conversation
                final_answer = await self._prepare_final_answer()
                logger.info(
                        f"(final answer)",
                        f"\n{final_answer}",
                )
                # Add final answer to chat history                                                                                         
                self._append_oai_message({"role": "assistant", "content": final_answer}, "assistant", self, True)
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
                    logger.info(
                            "(thought) Replan counter exceeded... Terminating."
                    )
                    return None
                # Let's create a new plan
                else:
                    logger.info(
                            "(thought) Stalled.... Replanning..."
                    )
                    

                    # Update our plan.
                    await self._update_facts_and_plan()

                    # Reset chat history but preserve initial task
                    initial_task = self._oai_messages[self][0]
                    self._oai_messages[self] = [initial_task]
                    
                    # Reset all agents
                    for agent in self._agents:
                        await agent.reset()
                    # Send everyone the NEW plan
                    synthesized_prompt = self._get_synthesize_prompt(
                        self._task, self._team_description, self._facts, self._plan
                    )
                    
                    # Share new plan with all agents
                    for agent in self._agents:
                        await self.a_send(synthesized_prompt, agent)

                    logger.info(
                            f"(thought) New plan:\n{synthesized_prompt}"
                    )

                    synthesized_message = {"role": "assistant", "content": synthesized_prompt}
                    self._append_oai_message(synthesized_message, "assistant", self, True)

                    # Answer from this synthesized message
                    return await self._select_next_agent(synthesized_prompt)

        # Select next agent and send instruction
        next_agent_name = ledger_dict["next_speaker"]["answer"]
        for agent in self._agents:
            if agent.name == next_agent_name:
                instruction = ledger_dict["instruction_or_question"]["answer"]
                
                # Log the instruction
                logger.info(f"{self.name} (-> {next_agent_name}): {instruction}")
                
                # Update chat history
                self._append_oai_message({"role": "assistant", "content": instruction}, "assistant", self, True)
                
                # Send instruction directly to the agent
                await self.a_send(instruction, agent)
                return agent

        return None

    async def initiate_chat(
        self,
        message: Optional[str] = None,
        clear_history: bool = True,
        silent: Optional[bool] = False,
    ) -> ChatResult:
        """Start the orchestration process with an initial message/task."""
        # Reset state
        self._current_round = 0
        if clear_history:
            self._oai_messages.clear()
            for agent in self._agents:
                agent.reset()
            
        if message is None:
            message = await self.a_get_human_input("Please provide the task: ")
            
        # Initialize the first agent selection
        next_agent = await self._select_next_agent(message)
        
        # Continue orchestration until max rounds reached or no next agent
        while next_agent is not None and self._current_round < self._max_rounds:
            self._current_round += 1
            
            # Get response from current agent
            response = await next_agent.a_generate_reply(sender=self)
            
            if response:
                # Add response to chat history
                self._append_oai_message({"role": "user", "content": response}, "user", self, False)
                
                # Select next agent based on response
                next_agent = await self._select_next_agent(response)
                
            if self._current_round >= self._max_rounds:
                logger.info(f"Max rounds ({self._max_rounds}) reached. Terminating.")
                
        # Return chat result with all relevant info
        return ChatResult(
            chat_history=self._oai_messages[self],
            summary=None,  # Could add summary generation if needed
            cost=None,     # Could track costs if needed
            human_input=self._human_input
        )
