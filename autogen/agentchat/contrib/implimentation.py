from typing import Any, Dict, List, Optional, Union, Literal
import json
import logging
import time
import traceback
from autogen.agentchat import (
    Agent, 
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    AgentProxy,
    UserMessage,
    AssistantMessage,
    BroadcastMessage,
    ResetMessage,
    RequestReplyMessage,
    MessageContext,
    TopicId,
    OrchestrationEvent,
)
from autogen.agentchat.utils import message_content_to_str
from autogen.core.base import CancellationToken

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

logger = logging.getLogger(__name__)


class OrchestratorGroupChat(GroupChat):
    """A GroupChat subclass that implements the orchestrator functionality using GroupChat's built-in mechanisms"""
    
    def __init__(
        self,
        agents: List[Agent],
        messages: List[Dict],
        max_round: int = 20,
        admin_name: str = "Admin",
        func_call_filter: bool = True,
        system_messages: List[Dict] = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_MESSAGE}],
        closed_book_prompt: str = ORCHESTRATOR_CLOSED_BOOK_PROMPT,
        plan_prompt: str = ORCHESTRATOR_PLAN_PROMPT,
        synthesize_prompt: str = ORCHESTRATOR_SYNTHESIZE_PROMPT,
        ledger_prompt: str = ORCHESTRATOR_LEDGER_PROMPT,
        update_facts_prompt: str = ORCHESTRATOR_UPDATE_FACTS_PROMPT,
        update_plan_prompt: str = ORCHESTRATOR_UPDATE_PLAN_PROMPT,
        replan_prompt: str = ORCHESTRATOR_REPLAN_PROMPT,
        max_stalls_before_replan: int = 3,
        max_replans: int = 3,
        return_final_answer: bool = False,
    ):
        super().__init__(
            agents=agents,
            messages=messages,
            max_round=max_round,
            admin_name=admin_name,
            func_call_filter=func_call_filter
        )

        self._system_messages = system_messages
        self._closed_book_prompt = closed_book_prompt
        self._plan_prompt = plan_prompt
        self._synthesize_prompt = synthesize_prompt
        self._ledger_prompt = ledger_prompt
        self._update_facts_prompt = update_facts_prompt
        self._update_plan_prompt = update_plan_prompt
        self._replan_prompt = replan_prompt

        self._max_stalls_before_replan = max_stalls_before_replan
        self._stall_counter = 0
        self._max_replans = max_replans
        self._replan_counter = 0
        self._return_final_answer = return_final_answer

        # State tracking
        self._team_description = self._get_team_description()
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

    async def _initialize_task(self, task: str, cancellation_token: Optional[CancellationToken] = None) -> None:
        # called the first time a task is received
        self._task = task
        self._team_description = self._get_team_description()

        # Shallow-copy the conversation
        planning_conversation = [m for m in self._chat_history]

        # 1. GATHER FACTS
        # create a closed book task and generate a response and update the chat history
        planning_conversation.append(
            {"role": "user", "content": self._get_closed_book_prompt(self._task)}
        )
        response = await self.generate_reply(
            self._system_messages + planning_conversation,
            cancellation_token=cancellation_token
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
        response = await self.generate_reply(
            self._system_messages + planning_conversation,
            cancellation_token=cancellation_token
        )

        assert isinstance(response, str)
        self._plan = response


    async def _update_facts_and_plan(self, cancellation_token: Optional[CancellationToken] = None) -> None:
        # called when the orchestrator decides to replan

        # Shallow-copy the conversation
        planning_conversation = [m for m in self._chat_history]

        # Update the facts
        planning_conversation.append(
            {"role": "user", "content": self._get_update_facts_prompt(self._task, self._facts)}
        )
        response = await self.generate_reply(
            self._system_messages + planning_conversation,
            cancellation_token=cancellation_token
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
        response = await self.generate_reply(
            self._system_messages + planning_conversation,
            cancellation_token=cancellation_token
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
            ledger_str = await self.generate_reply(
                self._system_messages + self._chat_history + ledger_user_messages,
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

                self.logger.error("An error occurred in update_ledger: %s", traceback.format_exc())

                self.logger.info(
                    OrchestrationEvent(
                        f"{self.metadata['type']} (error)",
                        f"Failed to parse ledger information: {ledger_str}",
                    )
                )
                raise e

        raise ValueError("Failed to parse ledger information after multiple retries.")

    async def _prepare_final_answer(self) -> str:
        # called when the task is complete

        final_message = {"role": "user", "content": ORCHESTRATOR_GET_FINAL_ANSWER.format(task=self._task)}
        response = await self.generate_reply(
            self._system_messages + self._chat_history + [final_message]
        )

        assert isinstance(response, str)
        return response


    async def _select_next_agent(
        self,
        task: str,
        cancellation_token: Optional[CancellationToken] = None
    ) -> Optional[AgentProxy]:
        # the main orchestrator loop
        # Check if the task is still unset, in which case this message contains the task string
        
        if len(self._task) == 0:
            await self._initialize_task(task, cancellation_token=cancellation_token)

            # At this point the task, plan and facts shouls all be set
            assert len(self._task) > 0
            assert len(self._facts) > 0
            assert len(self._plan) > 0
            assert len(self._team_description) > 0

            # Send everyone the plan
            synthesized_prompt = self._get_synthesize_prompt(
                self._task, self._team_description, self._facts, self._plan
            )
            self.chat_messages[self] = [{"role": "assistant", "content": synthesized_prompt}]
            
            #topic_id = TopicId("default", self.id.key)
            #await self.publish_message(
            #    BroadcastMessage(content=UserMessage(content=synthesized_prompt, source=self.metadata["type"])),
            #    topic_id=topic_id,
            #    cancellation_token=cancellation_token,
            #)

            self.logger.info(
                    f"Initial plan:\n{synthesized_prompt}"
            )

            self._replan_counter = 0
            self._stall_counter = 0

            synthesized_message = {"role": "assistant", "content": synthesized_prompt}
            self._chat_history.append(synthesized_message)

            # Answer from this synthesized message
            return await self._select_next_agent(synthesized_message, cancellation_token)

        # Orchestrate the next step
        ledger_dict = await self.update_ledger()
        self.logger.info(
                f"Updated Ledger:\n{json.dumps(ledger_dict, indent=2)}"
        )

        # Task is complete
        if ledger_dict["is_request_satisfied"]["answer"] is True:
            self.logger.info(
                    "Request satisfied."
            )
            if self._return_final_answer:
                # generate a final message to summarize the conversation
                final_answer = await self._prepare_final_answer()
                self.logger.info(
                        f"(final answer)",
                        f"\n{final_answer}",
                )
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
                    self.logger.info(
                            f"(thought)",
                            "Replan counter exceeded... Terminating.",
                    )
                    return None
                # Let's create a new plan
                else:
                    self.logger.info(
                        f"(thought)",
                            "Stalled.... Replanning...",
                        )
                    

                    # Update our plan.
                    await self._update_facts_and_plan()

                    # Reset everyone, then rebroadcast the new plan
                    self._chat_history = [self._chat_history[0]]
                    topic_id = TopicId("default", self.id.key)
                    await self.publish_message(ResetMessage(), topic_id=topic_id)

                    # Send everyone the NEW plan
                    synthesized_prompt = self._get_synthesize_prompt(
                        self._task, self._team_description, self._facts, self._plan
                    )
                    await self.publish_message(
                        BroadcastMessage(content=UserMessage(content=synthesized_prompt, source=self.metadata["type"])),
                        topic_id=topic_id,
                        cancellation_token=cancellation_token,
                    )

                    self.logger.info(
                            f"(thought)",
                            f"New plan:\n{synthesized_prompt}",
                    )

                    synthesized_message = {"role": "assistant", "content": synthesized_prompt}
                    self._chat_history.append(synthesized_message)

                    # Answer from this synthesized message
                    return await self._select_next_agent(synthesized_message, cancellation_token)

        # If we goit this far, we were not starting, done, or stuck
        next_agent_name = ledger_dict["next_speaker"]["answer"]
        # find the agent with the next agent name
        for agent in self._agents:
            if agent.name == next_agent_name:
                # broadcast a new message
                instruction = ledger_dict["instruction_or_question"]["answer"]
                user_message = {"role": "user", "content": instruction}
                assistant_message = {"role": "assistant", "content": instruction}
                self.logger.info(f"{self.name} (-> {next_agent_name})", instruction)
                self._chat_history.append(assistant_message)  # My copy
                topic_id = TopicId("default", self.id.key)
                await self.publish_message(
                    BroadcastMessage(content=user_message, request_halt=False),
                    topic_id=topic_id,
                    cancellation_token=cancellation_token,
                )  # Send to everyone else
                return agent

        return None



    async def _handle_broadcast(self, message: BroadcastMessage, ctx: MessageContext) -> None:
        """Handle an incoming message."""

        # First broadcast sets the timer
        if self._start_time < 0:
            self._start_time = time.time()

        source = "Unknown"
        if isinstance(message.content, UserMessage) or isinstance(message.content, AssistantMessage):
            source = message.content.source

        content = message_content_to_str(message.content.content)

        self.logger.info(OrchestrationEvent(source, content))

        # Termination conditions
        if self._num_rounds >= self._max_rounds:
            self.logger.info(
                OrchestrationEvent(
                    f"{self.metadata['type']} (termination condition)",
                    f"Max rounds ({self._max_rounds}) reached.",
                )
            )
            return

        if time.time() - self._start_time >= self._max_time:
            self.logger.info(
                OrchestrationEvent(
                    f"{self.metadata['type']} (termination condition)",
                    f"Max time ({self._max_time}s) reached.",
                )
            )
            return

        if message.request_halt:
            self.logger.info(
                OrchestrationEvent(
                    f"{self.metadata['type']} (termination condition)",
                    f"{source} requested halt.",
                )
            )
            return

        next_agent = await self._select_next_agent(message.content)
        if next_agent is None:
            self.logger.info(
                OrchestrationEvent(
                    f"{self.metadata['type']} (termination condition)",
                    "No agent selected.",
                )
            )
            return
        request_reply_message = RequestReplyMessage()
        # emit an event

        self.logger.info(
            OrchestrationEvent(
                source=f"{self.metadata['type']} (thought)",
                message=f"Next speaker {(await next_agent.metadata)['type']}" "",
            )
        )

        self._num_rounds += 1  # Call before sending the message
        await self.send_message(request_reply_message, next_agent.id, cancellation_token=ctx.cancellation_token)

