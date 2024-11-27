import base64
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import time
import traceback
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union, cast,Callable,Literal  # Any, Callable, Dict, List, Literal, Tuple
from urllib.parse import quote_plus  # parse_qs, quote, unquote, urlparse, urlunparse

import aiofiles

from PIL import Image
from playwright._impl._errors import Error as PlaywrightError
from playwright._impl._errors import TimeoutError
from autogen.agentchat import ConversableAgent, Agent


# from playwright._impl._async_base.AsyncEventInfo
from playwright.async_api import BrowserContext, Download, Page, Playwright, async_playwright

from .websurfer_prompts import (
    DEFAULT_DESCRIPTION,
    SCREENSHOT_TOOL_SELECTION,
)

# TODO: Fix mdconvert (I think i saw a new pull request)
from .markdown_browser import MarkdownConverter  # type: ignore
from .utils import SentinelMeta
from .image import AGImage

#from ...utils import message_content_to_str

from .set_of_mark import add_set_of_mark
from .tool_definitions import (
    TOOL_CLICK,
    TOOL_HISTORY_BACK,
    TOOL_PAGE_DOWN,
    TOOL_PAGE_UP,
    TOOL_READ_PAGE_AND_ANSWER,
    # TOOL_SCROLL_ELEMENT_DOWN,
    # TOOL_SCROLL_ELEMENT_UP,
    TOOL_SLEEP,
    TOOL_SUMMARIZE_PAGE,
    #TOOL_TYPE,
    TOOL_VISIT_URL,
    TOOL_WEB_SEARCH,
)

from .types import (
    InteractiveRegion,
    VisualViewport,
    interactiveregion_from_dict,
    visualviewport_from_dict,
)

# Viewport dimensions
VIEWPORT_HEIGHT = 900
VIEWPORT_WIDTH = 1440

# Size of the image we send to the MLM
# Current values represent a 0.85 scaling to fit within the GPT-4v short-edge constraints (768px)
MLM_HEIGHT = 765
MLM_WIDTH = 1224

SCREENSHOT_TOKENS = 1105

logger = logging.getLogger(__name__)

# Sentinels
class DEFAULT_CHANNEL(metaclass=SentinelMeta):
    pass


#@default_subscription
class MultimodalWebSurfer(ConversableAgent):
    """(In preview) A multimodal agent that acts as a web surfer that can search the web and visit web pages."""

    DEFAULT_START_PAGE = "https://www.bing.com/"

    def __init__(
        self,
        name: str = "MultimodalWebSurfer",
        system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = DEFAULT_DESCRIPTION,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        silent: Optional[bool] = None,
        screenshot_tool_prompt: str = SCREENSHOT_TOOL_SELECTION,
    ):
        """To instantiate properly please make sure to call MultimodalWebSurfer.init"""
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
            chat_messages=chat_messages,
            silent=silent
        )
        # Call init to set these
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._last_download: Download | None = None
        self._prior_metadata_hash: str | None = None
        #self.logger = logging.getLogger(EVENT_LOGGER_NAME + f".{self.id.key}.MultimodalWebSurfer")

        # Read page_script
        self._page_script: str = ""
        with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js"), "rt") as fh:
            self._page_script = fh.read()

        # Define the download handler
        def _download_handler(download: Download) -> None:
            self._last_download = download

        self._download_handler = _download_handler

        self.screenshot_tool_prompt = screenshot_tool_prompt

    async def init(
        self,
        headless: bool = True,
        browser_channel: str | type[DEFAULT_CHANNEL] = DEFAULT_CHANNEL,
        browser_data_dir: str | None = None,
        start_page: str | None = None,
        downloads_folder: str | None = None,
        debug_dir: str | None = os.getcwd(),
        to_save_screenshots: bool = False,
        # navigation_allow_list=lambda url: True,
        markdown_converter: Any | None = None,  # TODO: Fixme
    ) -> None:
        """
        Initialize the MultimodalWebSurfer.

        Args:
            headless (bool): Whether to run the browser in headless mode. Defaults to True.
            browser_channel (str | type[DEFAULT_CHANNEL]): The browser channel to use. Defaults to DEFAULT_CHANNEL.
            browser_data_dir (str | None): The directory to store browser data. Defaults to None.
            start_page (str | None): The initial page to visit. Defaults to DEFAULT_START_PAGE.
            downloads_folder (str | None): The folder to save downloads. Defaults to None.
            debug_dir (str | None): The directory to save debug information. Defaults to the current working directory.
            to_save_screenshots (bool): Whether to save screenshots. Defaults to False.
            markdown_converter (Any | None): The markdown converter to use. Defaults to None.
        """
        #self._model_client = model_client
        self.start_page = start_page or self.DEFAULT_START_PAGE
        self.downloads_folder = downloads_folder
        self.to_save_screenshots = to_save_screenshots
        self._chat_history: List[Dict[str,Any]] = [] # TODO: use the ag2 message format
        self._last_download = None
        self._prior_metadata_hash = None

        ## Create or use the provided MarkdownConverter
        if markdown_converter is None:
            self._markdown_converter = MarkdownConverter()  # type: ignore
        else:
            self._markdown_converter = markdown_converter  # type: ignore

        # Create the playwright self
        launch_args: Dict[str, Any] = {"headless": headless}
        if browser_channel is not DEFAULT_CHANNEL: 
            launch_args["channel"] = browser_channel
        self._playwright = await async_playwright().start()

        # Create the context -- are we launching persistent?
        if browser_data_dir is None:
            browser = await self._playwright.chromium.launch(**launch_args)
            self._context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
            )
        else:
            self._context = await self._playwright.chromium.launch_persistent_context(browser_data_dir, **launch_args)

        # Create the page
        self._context.set_default_timeout(60000)  # One minute
        self._page = await self._context.new_page()
        assert self._page is not None
        # self._page.route(lambda x: True, self._route_handler)
        self._page.on("download", self._download_handler)
        await self._page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
        await self._page.add_init_script(
            path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js")
        )
        await self._page.goto(self.start_page)
        await self._page.wait_for_load_state()

        # Prepare the debug directory -- which stores the screenshots generated throughout the process
        await self._set_debug_dir(debug_dir)

    def _get_screenshot_selection_prompt(self, page_url, visible_targets, other_targets_str, focused_hint, tool_names) -> str:
        assert self._page is not None
        return self.screenshot_tool_prompt.format(page_url=self._page.url,
        visible_targets=visible_targets,
        other_targets_str=other_targets_str,
        focused_hint=focused_hint,
        tool_names=tool_names
        )  

    async def _sleep(self, duration: Union[int, float]) -> None:
        assert self._page is not None
        await self._page.wait_for_timeout(duration * 1000)

    async def _set_debug_dir(self, debug_dir: str | None) -> None:
        assert self._page is not None
        self.debug_dir = debug_dir
        if self.debug_dir is None:
            return

        if not os.path.isdir(self.debug_dir):
            os.mkdir(self.debug_dir)
        current_timestamp = "_" + int(time.time()).__str__()
        screenshot_png_name = "screenshot" + current_timestamp + ".png"
        debug_html = os.path.join(self.debug_dir, "screenshot" + current_timestamp + ".html")
        if self.to_save_screenshots:
            async with aiofiles.open(debug_html, "wt") as file:
                await file.write(
                    f"""
    <html style="width:100%; margin: 0px; padding: 0px;">
    <body style="width: 100%; margin: 0px; padding: 0px;">
        <img src= {screenshot_png_name} id="main_image" style="width: 100%; max-width: {VIEWPORT_WIDTH}px; margin: 0px; padding: 0px;">
        <script language="JavaScript">
    var counter = 0;
    setInterval(function() {{
    counter += 1;
    document.getElementById("main_image").src = "screenshot.png?bc=" + counter;
    }}, 300);
        </script>
    </body>
    </html>
    """.strip(),
                )
        if self.to_save_screenshots:
            await self._page.screenshot(path=os.path.join(self.debug_dir, screenshot_png_name))
            logger.info(f"url: {self._page.url} screenshot: {screenshot_png_name}")

            logger.info(f"Multimodal Web Surfer debug screens: {pathlib.Path(os.path.abspath(debug_html)).as_uri()}\n")


    async def _reset(self) -> None: # TODO: ag2
        assert self._page is not None
        #future = super()._reset(cancellation_token) # TODO: ag2
        #await future
        await self._visit_page(self.start_page)
        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot" + current_timestamp + ".png"
            await self._page.screenshot(path=os.path.join(self.debug_dir, screenshot_png_name))  # type: ignore
            
            logger.info(f"url: {self._page.url} screenshot: {screenshot_png_name}")


        logger.info(f"url: {self._page.url} Resetting browser.")

    def _target_name(self, target: str, rects: Dict[str, InteractiveRegion]) -> str | None:
        try:
            return rects[target]["aria_name"].strip()
        except KeyError:
            return None

    def _format_target_list(self, ids: List[str], rects: Dict[str, InteractiveRegion]) -> List[str]:
        targets: List[str] = []
        for r in list(set(ids)):
            if r in rects:
                # Get the role
                aria_role = rects[r].get("role", "").strip()
                if len(aria_role) == 0:
                    aria_role = rects[r].get("tag_name", "").strip()

                # Get the name
                aria_name = re.sub(r"[\n\r]+", " ", rects[r].get("aria_name", "")).strip()

                # What are the actions?
                actions = ['"click"']
                if rects[r]["role"] in ["textbox", "searchbox", "search"]:
                    actions = ['"input_text"']
                actions_str = "[" + ",".join(actions) + "]"

                targets.append(f'{{"id": {r}, "name": "{aria_name}", "role": "{aria_role}", "tools": {actions_str} }}')

        return targets

    async def _generate_reply(self):# -> Tuple[bool, UserContent]: # TODO: ag2
        assert self._page is not None
        try:
            request_halt, content = await self.__generate_reply()
            return request_halt, content
        except Exception:
            return False, f"Web surfing error:\n\n{traceback.format_exc()}"

    async def _execute_tool( # TODO: ag2 biggest work integrating with tools
        self,
        message,#: List[FunctionCall],
        rects: Dict[str, InteractiveRegion],
        tool_names: str,
        use_ocr: bool = True
        ):# -> Tuple[bool, UserContent]:
        name = message[0].name
        args = json.loads(message[0].arguments)
        action_description = ""
        assert self._page is not None
        logger.info(f"url: {self._page.url} name: {name} args: {args} message: {f'{name}( {json.dumps(args)} )'}")

        if name == "visit_url":
            url = args.get("url")
            action_description = f"I typed '{url}' into the browser address bar."
            # Check if the argument starts with a known protocol
            if url.startswith(("https://", "http://", "file://", "about:")):
                await self._visit_page(url)
            # If the argument contains a space, treat it as a search query
            elif " " in url:
                await self._visit_page(f"https://www.bing.com/search?q={quote_plus(url)}&FORM=QBLH")
            # Otherwise, prefix with https://
            else:
                await self._visit_page("https://" + url)

        elif name == "history_back":
            action_description = "I clicked the browser back button."
            await self._back()

        elif name == "web_search":
            query = args.get("query")
            action_description = f"I typed '{query}' into the browser search bar."
            await self._visit_page(f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH")

        elif name == "page_up":
            action_description = "I scrolled up one page in the browser."
            await self._page_up()

        elif name == "page_down":
            action_description = "I scrolled down one page in the browser."
            await self._page_down()

        elif name == "click":
            target_id = str(args.get("target_id"))
            target_name = self._target_name(target_id, rects)
            if target_name:
                action_description = f"I clicked '{target_name}'."
            else:
                action_description = "I clicked the control."
            await self._click_id(target_id)

        elif name == "input_text":
            input_field_id = str(args.get("input_field_id"))
            text_value = str(args.get("text_value"))
            input_field_name = self._target_name(input_field_id, rects)
            if input_field_name:
                action_description = f"I typed '{text_value}' into '{input_field_name}'."
            else:
                action_description = f"I input '{text_value}'."
            await self._fill_id(input_field_id, text_value)

        elif name == "scroll_element_up":
            target_id = str(args.get("target_id"))
            target_name = self._target_name(target_id, rects)

            if target_name:
                action_description = f"I scrolled '{target_name}' up."
            else:
                action_description = "I scrolled the control up."

            await self._scroll_id(target_id, "up")

        elif name == "scroll_element_down":
            target_id = str(args.get("target_id"))
            target_name = self._target_name(target_id, rects)

            if target_name:
                action_description = f"I scrolled '{target_name}' down."
            else:
                action_description = "I scrolled the control down."

            await self._scroll_id(target_id, "down")

        elif name == "answer_question":
            question = str(args.get("question"))
            # Do Q&A on the DOM. No need to take further action. Browser state does not change.
            return False, await self._summarize_page(question=question)

        elif name == "summarize_page":
            # Summarize the DOM. No need to take further action. Browser state does not change.
            return False, await self._summarize_page()

        elif name == "sleep":
            action_description = "I am waiting a short period of time before taking further action."
            await self._sleep(3)  # There's a 2s sleep below too

        else:
            raise ValueError(f"Unknown tool '{name}'. Please choose from:\n\n{tool_names}")

        await self._page.wait_for_load_state()
        await self._sleep(3)

        # Handle downloads
        if self._last_download is not None and self.downloads_folder is not None:
            fname = os.path.join(self.downloads_folder, self._last_download.suggested_filename)
            # TOODO: Fix this type
            await self._last_download.save_as(fname)  # type: ignore
            page_body = f"<html><head><title>Download Successful</title></head><body style=\"margin: 20px;\"><h1>Successfully downloaded '{self._last_download.suggested_filename}' to local path:<br><br>{fname}</h1></body></html>"
            await self._page.goto(
                "data:text/html;base64," + base64.b64encode(page_body.encode("utf-8")).decode("utf-8")
            )
            await self._page.wait_for_load_state()

        # Handle metadata
        page_metadata = json.dumps(await self._get_page_metadata(), indent=4)
        metadata_hash = hashlib.md5(page_metadata.encode("utf-8")).hexdigest()
        if metadata_hash != self._prior_metadata_hash:
            page_metadata = (
                "\nThe following metadata was extracted from the webpage:\n\n" + page_metadata.strip() + "\n"
            )
        else:
            page_metadata = ""
        self._prior_metadata_hash = metadata_hash

        # Describe the viewport of the new page in words
        viewport = await self._get_visual_viewport()
        percent_visible = int(viewport["height"] * 100 / viewport["scrollHeight"])
        percent_scrolled = int(viewport["pageTop"] * 100 / viewport["scrollHeight"])
        if percent_scrolled < 1:  # Allow some rounding error
            position_text = "at the top of the page"
        elif percent_scrolled + percent_visible >= 99:  # Allow some rounding error
            position_text = "at the bottom of the page"
        else:
            position_text = str(percent_scrolled) + "% down from the top of the page"

        new_screenshot = await self._page.screenshot()
        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot" + current_timestamp + ".png"
            async with aiofiles.open(os.path.join(self.debug_dir, screenshot_png_name), "wb") as file:  # type: ignore
                await file.write(new_screenshot)  # type: ignore
            logger.info(f"url: {self._page.url} Screenshot: {screenshot_png_name}")
            #self.logger.info(
            #    WebSurferEvent(
            #        source=self.metadata["type"],
            #        url=self._page.url,
            #        message="Screenshot: " + screenshot_png_name,
            #    )
            #)

        ocr_text = (
            await self._get_ocr_text(new_screenshot) if use_ocr is True else ""
        )

        # Return the complete observation
        message_content = ""  # message.content or ""
        page_title = await self._page.title()

        return False, [
            f"{message_content}\n\n{action_description}\n\nHere is a screenshot of [{page_title}]({self._page.url}). The viewport shows {percent_visible}% of the webpage, and is positioned {position_text}.{page_metadata}\nAutomatic OCR of the page screenshot has detected the following text:\n\n{ocr_text}".strip(),
            AGImage.from_pil(Image.open(io.BytesIO(new_screenshot))),
        ]

    async def __generate_reply(self):# -> Tuple[bool, UserContent]: # TODO: ag2
        assert self._page is not None
        """Generates the actual reply. First calls the LLM to figure out which tool to use, then executes the tool."""

        # Clone the messages to give context, removing old screenshots
        # TODO: massage types in ag2 ? 
        history: List[Dict[str,Any]] = [] 
        for m in self._chat_history:
            if isinstance(m.content, str):
                history.append(m)
            elif isinstance(m.content, list):
                content = message_content_to_str(m.content)
                if isinstance(m, UserMessage):
                    history.append(UserMessage(content=content, source=m.source))
                elif isinstance(m, AssistantMessage):
                    history.append(AssistantMessage(content=content, source=m.source))
                elif isinstance(m, SystemMessage):
                    history.append(SystemMessage(content=content))

        # Ask the page for interactive elements, then prepare the state-of-mark screenshot
        rects = await self._get_interactive_rects() 
        viewport = await self._get_visual_viewport()
        screenshot = await self._page.screenshot()
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(screenshot, rects)

        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot_som" + current_timestamp + ".png"
            som_screenshot.save(os.path.join(self.debug_dir, screenshot_png_name))  # type: ignore
            self.logger.info(
                WebSurferEvent(
                    source=self.metadata["type"],
                    url=self._page.url,
                    message="Screenshot: " + screenshot_png_name,
                )
            )
        # What tools are available?
        tools = [
            TOOL_VISIT_URL,
            TOOL_HISTORY_BACK,
            TOOL_CLICK,
            TOOL_TYPE,
            TOOL_SUMMARIZE_PAGE,
            TOOL_READ_PAGE_AND_ANSWER,
            TOOL_SLEEP,
        ]

        # Can we reach Bing to search?
        # if self._navigation_allow_list("https://www.bing.com/"):
        tools.append(TOOL_WEB_SEARCH)

        # We can scroll up
        if viewport["pageTop"] > 5:
            tools.append(TOOL_PAGE_UP)

        # Can scroll down
        if (viewport["pageTop"] + viewport["height"] + 5) < viewport["scrollHeight"]:
            tools.append(TOOL_PAGE_DOWN)

        # Focus hint
        focused = await self._get_focused_rect_id()
        focused_hint = ""
        if focused:
            name = self._target_name(focused, rects)
            if name:
                name = f"(and name '{name}') "

            role = "control"
            try:
                role = rects[focused]["role"]
            except KeyError:
                pass

            focused_hint = f"\nThe {role} with ID {focused} {name}currently has the input focus.\n\n"

        # Everything visible
        visible_targets = "\n".join(self._format_target_list(visible_rects, rects)) + "\n\n"

        # Everything else
        other_targets: List[str] = []
        other_targets.extend(self._format_target_list(rects_above, rects))
        other_targets.extend(self._format_target_list(rects_below, rects))

        if len(other_targets) > 0:
            other_targets_str = (
                "Additional valid interaction targets (not shown) include:\n" + "\n".join(other_targets) + "\n\n"
            )
        else:
            other_targets_str = ""

        # If there are scrollable elements, then add the corresponding tools
        # has_scrollable_elements = False
        # if has_scrollable_elements:
        #    tools.append(TOOL_SCROLL_ELEMENT_UP)
        #    tools.append(TOOL_SCROLL_ELEMENT_DOWN)

        tool_names = "\n".join([t["name"] for t in tools])

        text_prompt = self._get_screenshot_selection_prompt(self._page.url, 
        visible_targets, 
        other_targets_str, 
        focused_hint, 
        tool_names)

        # Scale the screenshot for the MLM, and close the original
        scaled_screenshot = som_screenshot.resize((MLM_WIDTH, MLM_HEIGHT))
        som_screenshot.close()
        if self.to_save_screenshots:
            scaled_screenshot.save(os.path.join(self.debug_dir, "screenshot_scaled.png"))  # type: ignore

        # Add the multimodal message and make the request
        history.append(
            {
                "role": "user",
                "content": [text_prompt, AGImage.from_pil(scaled_screenshot)],
            }
        )
        self.chat_messages.append(history)
        response = self.generate_reply(history) # TODO: make sure tool are added
        #response = await self._model_client.create( # TODO: ag2 / generate reply ?
        #    history, tools=tools, extra_create_args={"tool_choice": "auto"}, cancellation_token=cancellation_token
        #)  # , "parallel_tool_calls": False})
        message = response

        self._last_download = None

        if isinstance(message, str):
            # Answer directly
            return False, message
        elif isinstance(message, dict):
            # Take an action
            self.generate_tool_calls_reply(message)
            return await self._execute_tool(message, rects, tool_names, cancellation_token=cancellation_token)
        else:
            # Not sure what happened here
            raise AssertionError(f"Unknown response format '{message}'")

    async def _get_interactive_rects(self) -> Dict[str, InteractiveRegion]:
        assert self._page is not None

        # Read the regions from the DOM
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass
        result = cast(
            Dict[str, Dict[str, Any]], await self._page.evaluate("MultimodalWebSurfer.getInteractiveRects();")
        )

        # Convert the results into appropriate types
        assert isinstance(result, dict)
        typed_results: Dict[str, InteractiveRegion] = {}
        for k in result:
            assert isinstance(k, str)
            typed_results[k] = interactiveregion_from_dict(result[k])

        return typed_results

    async def _get_visual_viewport(self) -> VisualViewport:
        assert self._page is not None
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass
        return visualviewport_from_dict(await self._page.evaluate("MultimodalWebSurfer.getVisualViewport();"))

    async def _get_focused_rect_id(self) -> str:
        assert self._page is not None
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass
        result = await self._page.evaluate("MultimodalWebSurfer.getFocusedElementId();")
        return str(result)

    async def _get_page_metadata(self) -> Dict[str, Any]:
        assert self._page is not None
        try:
            await self._page.evaluate(self._page_script)
        except Exception:
            pass
        result = await self._page.evaluate("MultimodalWebSurfer.getPageMetadata();")
        assert isinstance(result, dict)
        return cast(Dict[str, Any], result)

    async def _get_page_markdown(self) -> str:
        assert self._page is not None
        html = await self._page.evaluate("document.documentElement.outerHTML;")
        # TOODO: fix types
        res = self._markdown_converter.convert_stream(io.StringIO(html), file_extension=".html", url=self._page.url)  # type: ignore
        return res.text_content  # type: ignore

    async def _on_new_page(self, page: Page) -> None:
        self._page = page
        assert self._page is not None
        # self._page.route(lambda x: True, self._route_handler)
        self._page.on("download", self._download_handler)
        await self._page.set_viewport_size({"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT})
        await self._sleep(0.2)
        self._prior_metadata_hash = None
        await self._page.add_init_script(
            path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js")
        )
        await self._page.wait_for_load_state()

    async def _back(self) -> None:
        assert self._page is not None
        await self._page.go_back()

    async def _visit_page(self, url: str) -> None:
        assert self._page is not None
        try:
            # Regular webpage
            await self._page.goto(url)
            await self._page.wait_for_load_state()
            self._prior_metadata_hash = None
        except Exception as e_outer:
            # Downloaded file
            if self.downloads_folder and "net::ERR_ABORTED" in str(e_outer):
                async with self._page.expect_download() as download_info:
                    try:
                        await self._page.goto(url)
                    except Exception as e_inner:
                        if "net::ERR_ABORTED" in str(e_inner):
                            pass
                        else:
                            raise e_inner
                    download = await download_info.value
                    fname = os.path.join(self.downloads_folder, download.suggested_filename)
                    await download.save_as(fname)
                    message = f"<body style=\"margin: 20px;\"><h1>Successfully downloaded '{download.suggested_filename}' to local path:<br><br>{fname}</h1></body>"
                    await self._page.goto(
                        "data:text/html;base64," + base64.b64encode(message.encode("utf-8")).decode("utf-8")
                    )
                    self._last_download = None  # Since we already handled it
            else:
                raise e_outer

    async def _page_down(self) -> None:
        assert self._page is not None
        await self._page.evaluate(f"window.scrollBy(0, {VIEWPORT_HEIGHT-50});")

    async def _page_up(self) -> None:
        assert self._page is not None
        await self._page.evaluate(f"window.scrollBy(0, -{VIEWPORT_HEIGHT-50});")

    async def _click_id(self, identifier: str) -> None:
        assert self._page is not None
        target = self._page.locator(f"[__elementId='{identifier}']")

        # See if it exists
        try:
            await target.wait_for(timeout=100)
        except TimeoutError:
            raise ValueError("No such element.") from None

        # Click it
        await target.scroll_into_view_if_needed()
        box = cast(Dict[str, Union[int, float]], await target.bounding_box())
        try:
            # Give it a chance to open a new page
            # TOODO: Having trouble with these types 
            async with self._page.expect_event("popup", timeout=1000) as page_info:  # type: ignore
                await self._page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, delay=10)
                # If we got this far without error, than a popup or new tab opened. Handle it.

                new_page = await page_info.value  # type: ignore

                assert isinstance(new_page, Page)
                await self._on_new_page(new_page)

                logger.info(f"url: {self._page.url} New tab or window.")

        except TimeoutError:
            pass

    async def _fill_id(self, identifier: str, value: str) -> None:
        assert self._page is not None
        target = self._page.locator(f"[__elementId='{identifier}']")

        # See if it exists
        try:
            await target.wait_for(timeout=100)
        except TimeoutError:
            raise ValueError("No such element.") from None

        # Fill it
        await target.scroll_into_view_if_needed()
        await target.focus()
        try:
            await target.fill(value)
        except PlaywrightError:
            await target.press_sequentially(value)
        await target.press("Enter")

    async def _scroll_id(self, identifier: str, direction: str) -> None:
        assert self._page is not None
        await self._page.evaluate(
            f"""
        (function() {{
            let elm = document.querySelector("[__elementId='{identifier}']");
            if (elm) {{
                if ("{direction}" == "up") {{
                    elm.scrollTop = Math.max(0, elm.scrollTop - elm.clientHeight);
                }}
                else {{
                    elm.scrollTop = Math.min(elm.scrollHeight - elm.clientHeight, elm.scrollTop + elm.clientHeight);
                }}
            }}
        }})();
    """
        )

    async def _summarize_page(
        self,
        question: str | None = None,
        token_limit: int = 100000
    ) -> str:
        assert self._page is not None

        page_markdown: str = await self._get_page_markdown()

        title: str = self._page.url
        try:
            title = await self._page.title()
        except Exception:
            pass

        # Take a screenshot and scale it
        screenshot = Image.open(io.BytesIO(await self._page.screenshot()))
        scaled_screenshot = screenshot.resize((MLM_WIDTH, MLM_HEIGHT))
        screenshot.close()
        ag_image = AGImage.from_pil(scaled_screenshot)

        # Prepare the system prompt
        messages: List[Dict[str,Any]] = []
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant that can summarize long documents to answer question.",
            }
        )

        # Prepare the main prompt
        prompt = f"We are visiting the webpage '{title}'. Its full-text content are pasted below, along with a screenshot of the page's current viewport."
        if question is not None:
            prompt += f" Please summarize the webpage into one or two paragraphs with respect to '{question}':\n\n"
        else:
            prompt += " Please summarize the webpage into one or two paragraphs:\n\n"

        # Grow the buffer (which is added to the prompt) until we overflow the context window or run out of lines
        buffer = ""
        for line in re.split(r"([\r\n]+)", page_markdown):
            message = {
                "role": "user",
                "content": [prompt + 
                buffer + 
                line, 
                #ag_image,
                ],
            }
            
            #UserMessage(
                # content=[
                #prompt + buffer + line,
                #    ag_image,
                # ],
                #source=self.metadata["type"],
            #)

            # TODO: is something like this possible in ag2
            #remaining = self._model_client.remaining_tokens(messages + [message])
            #if remaining > SCREENSHOT_TOKENS:
            #    buffer += line
            #else:
            #    break

        # Nothing to do
        buffer = buffer.strip()
        if len(buffer) == 0:
            return "Nothing to summarize."

        # Append the message
        messages.append(
            {
                "role": "user",
                "content": [prompt + buffer, ag_image],
            }
        )

        # Generate the response
        is_valid_response, response = self.generate_oai_reply(messages=messages)
        scaled_screenshot.close()
        assert is_valid_response
        assert isinstance(response, str)
        return response

    async def _get_ocr_text(
        self, image: bytes | io.BufferedIOBase | Image.Image
    ) -> str:
        scaled_screenshot = None
        if isinstance(image, Image.Image):
            scaled_screenshot = image.resize((MLM_WIDTH, MLM_HEIGHT))
        else:
            pil_image = None
            if not isinstance(image, io.BufferedIOBase):
                pil_image = Image.open(io.BytesIO(image))
            else:
                # TOODO: Not sure why this cast was needed, but by this point screenshot is a binary file-like object
                pil_image = Image.open(cast(BinaryIO, image))
            scaled_screenshot = pil_image.resize((MLM_WIDTH, MLM_HEIGHT))
            pil_image.close()

        messages: List[Dict[str,Any]] = [] 
        messages.append(
            {"role": "user", 
            "content": ["Please transcribe all visible text on this page, including both main content and the labels of UI elements.",
                        AGImage.from_pil(scaled_screenshot)]}
        )
        is_valid_response, response = self.generate_oai_reply(messages=messages)

        scaled_screenshot.close()
        assert is_valid_response
        assert isinstance(response, str)
        return response
