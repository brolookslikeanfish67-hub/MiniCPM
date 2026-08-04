import ast
import json
import re
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

# Define structural schemas for vLLM integration
class FunctionCall:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class ToolCall:
    def __init__(self, id: str, type: str, function: FunctionCall):
        self.id = id
        self.type = type
        self.function = function


class MiniCPM5XMLToolParser:
    """XML Tool Parser for MiniCPM-5 function calling in vLLM."""

    # Pre-compiled regex patterns for performance
    FUNCTION_PATTERN = re.compile(
        r'<function\s+name=["\'](?P<name>[^"\']+)["\']\s*>(?P<body>.*?)</function>',
        re.DOTALL | re.IGNORECASE,
    )
    PARAM_PATTERN = re.compile(
        r'<param\s+name=["\'](?P<pname>[^"\']+)["\']\s*>(?P<pval>.*?)</param>',
        re.DOTALL | re.IGNORECASE,
    )
    SINGLE_TAG_PATTERN = re.compile(
        r'<param\s+name=["\'](?P<pname>[^"\']+)["\']\s+value=["\'](?P<pval>[^"\']+)["\']\s*/>',
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self):
        self.reset_streaming_state()

    def reset_streaming_state(self) -> None:
        """Reset internal buffer state for streaming tool call generation."""
        self._stream_buffer: str = ""
        self._current_tool_id: Optional[str] = None
        self._current_func_name: Optional[str] = None
        self._parsed_args: Dict[str, Any] = {}
        self._in_tool_call: bool = False

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normalize token space artifacts and minor XML malformations."""
        if not text:
            return ""

        # Remove SentencePiece / GPT token artifacts (e.g., \u0120 or 'Ġ')
        normalized = text.replace("\u0120", " ").replace("Ġ", " ")

        # Clean unclosed quotes in standard attribute setups
        normalized = re.sub(r'name=\s*"([^"]*)$', r'name="\1"', normalized)
        return normalized

    def _safe_parse_value(self, val_str: str) -> Any:
        """Attempt multi-stage parsing for XML parameter values."""
        val_str = val_str.strip()
        if not val_str:
            return ""

        # Stage 1: Try JSON parsing (handles booleans, numbers, lists, dicts, strings)
        try:
            return json.loads(val_str)
        except json.JSONDecodeError:
            pass

        # Stage 2: Try Literal evaluation for Python structures
        try:
            return ast.literal_eval(val_str)
        except (ValueError, SyntaxError):
            pass

        # Stage 3: Return raw string
        return val_str

    def extract_tool_calls(self, text: str) -> Tuple[str, List[ToolCall]]:
        """
        Parse complete model text output into non-tool content and extracted ToolCall objects.
        """
        normalized_text = self.normalize_text(text)
        tool_calls: List[ToolCall] = []
        last_end = 0
        content_parts: List[str] = []

        for index, match in enumerate(self.FUNCTION_PATTERN.finditer(normalized_text)):
            start, end = match.span()
            content_parts.append(normalized_text[last_end:start])

            func_name = match.group("name")
            func_body = match.group("body")
            args_dict: Dict[str, Any] = {}

            # Process paired <param> tags
            for param_match in self.PARAM_PATTERN.finditer(func_body):
                p_name = param_match.group("pname")
                p_val_raw = param_match.group("pval")
                args_dict[p_name] = self._safe_parse_value(p_val_raw)

            # Process self-closing <param /> tags
            for param_match in self.SINGLE_TAG_PATTERN.finditer(func_body):
                p_name = param_match.group("pname")
                p_val_raw = param_match.group("pval")
                if p_name not in args_dict:
                    args_dict[p_name] = self._safe_parse_value(p_val_raw)

            tool_call = ToolCall(
                id=f"call_{index}_{func_name}",
                type="function",
                function=FunctionCall(
                    name=func_name,
                    arguments=json.dumps(args_dict, ensure_ascii=False),
                ),
            )
            tool_calls.append(tool_call)
            last_end = end

        content_parts.append(normalized_text[last_end:])
        cleaned_content = "".join(content_parts).strip()

        return cleaned_content, tool_calls

    def extract_tool_calls_streaming(
        self, token: str
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process incremental token chunks and yield streaming deltas or raw content chunks.
        """
        normalized_token = self.normalize_text(token)
        self._stream_buffer += normalized_token

        # Check for start of tool invocation if not already inside one
        if not self._in_tool_call:
            if "<function" in self._stream_buffer:
                tag_start = self._stream_buffer.find("<function")
                # Yield text preceding the tag
                if tag_start > 0:
                    yield {"type": "content", "content": self._stream_buffer[:tag_start]}
                    self._stream_buffer = self._stream_buffer[tag_start:]

                self._in_tool_call = True
            else:
                # Flush buffer if no partial XML tag is forming
                if not any("<function".startswith(self._stream_buffer[-i:]) for i in range(1, len("<function"))):
                    yield {"type": "content", "content": self._stream_buffer}
                    self._stream_buffer = ""
                return

        # Complete tag extraction when closing tag is encountered
        if "</function>" in self._stream_buffer:
            end_tag_idx = self._stream_buffer.find("</function>") + len("</function>")
            complete_xml = self._stream_buffer[:end_tag_idx]
            self._stream_buffer = self._stream_buffer[end_tag_idx:]

            _, tool_calls = self.extract_tool_calls(complete_xml)
            for tool_call in tool_calls:
                yield {"type": "tool_call", "tool_call": tool_call}

            self.reset_streaming_state()
