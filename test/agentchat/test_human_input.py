# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3 -m pytest

from unittest.mock import MagicMock

import pytest

import autogen

from ..conftest import reason, skip_openai  # noqa: E402
from .test_assistant_agent import KEY_LOC, OAI_CONFIG_LIST


@pytest.mark.skipif(skip_openai, reason=reason)
def test_get_human_input():
    config_list = autogen.config_list_from_json(OAI_CONFIG_LIST, KEY_LOC, filter_dict={"tags": ["gpt-4o-mini"]})

    # create an AssistantAgent instance named "assistant"
    assistant = autogen.AssistantAgent(
        name="assistant",
        max_consecutive_auto_reply=2,
        llm_config={"timeout": 600, "cache_seed": 41, "config_list": config_list, "temperature": 0},
    )

    user_proxy = autogen.UserProxyAgent(name="user", human_input_mode="ALWAYS", code_execution_config=False)

    # Use MagicMock to create a mock get_human_input function
    user_proxy.get_human_input = MagicMock(return_value="This is a test")

    res = user_proxy.initiate_chat(assistant, clear_history=True, message="Hello.")
    print("Result summary:", res.summary)
    print("Human input:", res.human_input)

    # Test without supplying messages parameter
    res = user_proxy.initiate_chat(assistant, clear_history=True)
    print("Result summary:", res.summary)
    print("Human input:", res.human_input)

    # Assert that custom_a_get_human_input was called at least once
    user_proxy.get_human_input.assert_called()


if __name__ == "__main__":
    test_get_human_input()
