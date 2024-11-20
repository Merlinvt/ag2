import json
import logging
import re
from typing import Dict, List, Optional, Any
from .orchestrator_prompts import (
    ORCHESTRATOR_CLOSED_BOOK_PROMPT,
    ORCHESTRATOR_PLAN_PROMPT,
    ORCHESTRATOR_LEDGER_PROMPT,
    ORCHESTRATOR_UPDATE_FACTS_PROMPT,
    ORCHESTRATOR_REPLAN_PROMPT
)

logger = logging.getLogger(__name__)

class OrchestratorPromptHandler:
    """Handles prompt generation and processing for the OrchestratorAgent."""
    
    def __init__(self, agent):
        """
        Initialize with reference to parent agent for LLM access.
        
        Args:
            agent: The parent OrchestratorAgent instance
        """
        self._agent = agent
        
    def generate_team_description(self, agents: List[Any]) -> str:
        """Generate a description of the team's capabilities."""
        descriptions = []
        for agent in agents:
            descriptions.append(agent.description)
        return "\n".join(descriptions)

    def get_synthesize_prompt(self, task: str, team: str, facts: str, plan: str) -> str:
        """Create a synthesis prompt combining task, team, facts, and plan."""
        return self._agent._synthesize_prompt.format(
            task=task,
            team=team,
            facts=facts,
            plan=plan
        )

    def analyze_task(self, task: str) -> str:
        """Analyze the task using the closed book prompt."""
        facts = self._agent.generate_reply(
            messages=[
                {"role": "system", "content": ORCHESTRATOR_CLOSED_BOOK_PROMPT.format(task=task)}
            ]
        )
        logger.info(f"Task analysis complete: {facts}")
        return facts

    def create_plan(self, team_description: str) -> str:
        """Create a plan using the plan prompt."""
        return self._agent.generate_reply(
            messages=[
                {"role": "system", "content": ORCHESTRATOR_PLAN_PROMPT.format(
                    team_description=team_description
                )}
            ]
        )

    def update_facts_and_plan(self, state: Dict[str, Any], team_description: str) -> tuple[str, str]:
        """Update facts and plan based on current state."""
        # Update facts
        new_facts = self._agent.generate_reply(
            messages=[{
                "role": "system",
                "content": ORCHESTRATOR_UPDATE_FACTS_PROMPT.format(
                    task=state["task"],
                    previous_facts=state["facts"] if state["facts"] else "No previous facts available."
                )
            }]
        )

        # Create replan prompt with current state
        replan_prompt = ORCHESTRATOR_REPLAN_PROMPT.format(
            reason="conversation stalled or in loop",
            team_description=team_description,
            completed_steps=", ".join(state["completed_subtasks"]),
            failed_attempts=", ".join(f"{attempt['error_type']}" for attempt in state["failed_attempts"]),
            blockers="conversation stalled or in loop"
        )

        # Get new plan using replan prompt
        new_plan = self._agent.generate_reply(
            messages=[{
                "role": "system",
                "content": replan_prompt
            }]
        )
        
        return new_facts, new_plan

    def update_ledger(self, task: str, team_description: str, agent_roles: List[str], chat_messages: Dict) -> Dict[str, Any]:
        """Update and return the ledger state."""
        max_retries = 3
        ledger_prompt = ORCHESTRATOR_LEDGER_PROMPT.format(
            task=task,
            team_description=team_description,
            agent_roles=json.dumps(agent_roles)
        )

        for _ in range(max_retries):
            try:
                messages = []
                
                # Add chat messages if they exist
                if chat_messages:
                    for agent_messages in chat_messages.values():
                        for msg in agent_messages:
                            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                messages.append(msg)
                
                # Add ledger prompt as final assistant message
                messages.append({
                    "role": "assistant", 
                    "content": ledger_prompt
                })
                
                response = self._agent.generate_reply(messages=messages)
                ledger = self._clean_and_parse_json(response)
                
                if self._validate_ledger(ledger):
                    return ledger
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("An error occurred: %s", e, exc_info=True)
                continue
                
        raise ValueError("Failed to get valid ledger after multiple retries")

    def _clean_and_parse_json(self, content: str) -> Dict:
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
                raise ValueError("No JSON structure found in content")
            content = json_match.group(0)

        # Preserve newlines for readability in error messages
        formatted_content = content
        
        # Now clean for parsing
        try:
            # First try parsing the cleaned but formatted content
            return json.loads(formatted_content)
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
                logger.error(f"Original content:\n{formatted_content}")
                logger.error(f"Cleaned content:\n{cleaned_content}")
                logger.error(f"JSON error: {str(e)}")
                raise ValueError(f"Failed to parse JSON after cleaning. Error: {str(e)}")

    def _validate_ledger(self, ledger: Dict) -> bool:
        """Validate ledger structure and content."""
        required_keys = [
            "is_request_satisfied",
            "is_in_loop", 
            "is_progress_being_made",
            "next_speaker",
            "instruction_or_question"
        ]
        
        if not isinstance(ledger, dict):
            logger.error(f"Ledger must be a dictionary, got {type(ledger)}")
            return False
        
        for key in required_keys:
            if key not in ledger:
                logger.error(f"Missing required key '{key}' in ledger")
                return False
                
            if not isinstance(ledger[key], dict):
                logger.error(f"Value for '{key}' must be a dictionary, got {type(ledger[key])}")
                return False
                
            for subfield in ["answer", "reason"]:
                if subfield not in ledger[key]:
                    logger.error(f"Missing required subfield '{subfield}' in ledger['{key}']")
                    return False
                    
                # Allow null/None values for answer field
                if subfield == "answer":
                    if not isinstance(ledger[key][subfield], (str, bool, type(None))):
                        logger.error(f"Answer field must be string, boolean or null, got {type(ledger[key][subfield])}")
                        return False
        
        return True
