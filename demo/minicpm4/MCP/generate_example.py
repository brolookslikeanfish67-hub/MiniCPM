import argparse
import ast
import json
import keyword
import re
import uuid
from typing import Any, Dict, List

import openai
from transformers import AutoTokenizer

# ==============================================================================
# AST Resolution & Parsing Utilities
# ==============================================================================

def resolve_ast_node(node: ast.AST) -> Any:
    """Recursively evaluates Python AST nodes into native Python objects."""
    if isinstance(node, ast.Constant):
        return "..." if node.value is Ellipsis else node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -resolve_ast_node(node.operand)
    elif isinstance(node, ast.List):
        return [resolve_ast_node(elt) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(resolve_ast_node(elt) for elt in node.elts)
    elif isinstance(node, ast.Dict):
        return {
            resolve_ast_node(k): resolve_ast_node(v)
            for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Call):
        if not node.keywords:
            return ast.unparse(node)
        return resolve_ast_call(node)
    elif isinstance(node, ast.Subscript):
        return ast.unparse(node)
    elif isinstance(node, (ast.BinOp, ast.Lambda)):
        return ast.unparse(node)
    else:
        return ast.unparse(node)


def resolve_ast_call(node: ast.Call) -> Dict[str, Dict[str, Any]]:
    """Extracts function name and keyword arguments from an ast.Call node."""
    # Resolve full qualified function name (e.g., module.submodule.func)
    func_parts = []
    curr = node.func
    while isinstance(curr, ast.Attribute):
        func_parts.append(curr.attr)
        curr = curr.value
    if isinstance(curr, ast.Name):
        func_parts.append(curr.id)
    
    func_name = ".".join(reversed(func_parts))

    kwargs = {
        kw.arg: resolve_ast_node(kw.value)
        for kw in node.keywords
        if kw.arg is not None
    }
    return {func_name: kwargs}


def extract_code_block(text: str, start_tag: str, end_tag: str) -> str:
    """Extracts raw tool call strings between tags and handles markdown wrapping."""
    if start_tag not in text or end_tag not in text:
        return ""
    
    raw = text.rsplit(end_tag, 1)[0].split(start_tag, 1)[1].strip()
    
    # Clean markdown formatting if present
    if raw.startswith("```"):
        raw = raw[3:].strip()
        if raw.startswith("python"):
            raw = raw[6:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
        
    return raw


def parse_tool_for_minicpm3(
    sequence: str,
    tool_call_start: str = "<|tool_call_start|>",
    tool_call_end: str = "<|tool_call_end|>",
) -> List[Dict[str, Any]]:
    """Parses tool calls embedded in a sequence string into OpenAI tool call specs."""
    try:
        raw_call_str = extract_code_block(sequence, tool_call_start, tool_call_end)
        if not raw_call_str:
            return []

        # Safe replacement of reserved Python keywords in kwarg signatures
        for kw in keyword.kwlist:
            raw_call_str = re.sub(
                rf'([,(]\s*){kw}(\s*=)',
                rf'\1{kw}_\2',
                raw_call_str
            )

        # Normalize hyphens in function names
        need_hyphen_restore = "-" in raw_call_str
        cleaned_str = raw_call_str.replace("-", "_")

        parsed_ast: ast.Module = ast.parse(cleaned_str)
        tool_calls = []

        for stmt in parsed_ast.body:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
                
            calls = resolve_ast_call(stmt.value)
            for func_name, func_args in calls.items():
                # Revert keyword mangling
                restored_args = {
                    (k[:-1] if k.endswith("_") and k[:-1] in keyword.kwlist else k): v
                    for k, v in func_args.items()
                }

                if need_hyphen_restore:
                    func_name = func_name.replace("_", "-")

                tool_calls.append({
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": restored_args
                    }
                })

        return tool_calls
    except Exception:
        return []

# ==============================================================================
# Model Interaction & History Normalization
# ==============================================================================

def generate_completion(
    client: openai.OpenAI,
    tokenizer: AutoTokenizer,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]]
) -> str:
    """Formats prompt with chat template and queries the model client."""
    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )

    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=8192
    )

    return response.choices[0].text


def format_conversation_history(
    messages: List[Dict[str, Any]],
    generate_result: str,
    new_tool_calls: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Converts standard chat messages into target serialized log format."""
    formatted_history = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            formatted_history.append({"system": content})
        elif role == "user":
            formatted_history.append({"human": content})
        elif role == "assistant":
            history_calls = parse_tool_for_minicpm3(content)
            formatted_history.append({
                "gpt": content,
                "function_call": history_calls
            })
        elif role == "tool":
            if formatted_history and "observation" in formatted_history[-1]:
                formatted_history[-1]["observation"].append(content)
            elif formatted_history:
                formatted_history[-1]["observation"] = [content]

    formatted_history.append({
        "gpt": generate_result,
        "function_call": new_tool_calls
    })

    return formatted_history

# ==============================================================================
# Main Pipeline
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="MiniCPM3 Tool Call Pipeline")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    client = openai.OpenAI(api_key="none", base_url=args.base_url)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    with open("available_tool_example.json", "r", encoding="utf-8") as f:
        available_tools = json.load(f)["available_tools"]

    # Generate response from model
    generate_result = generate_completion(
        client=client,
        tokenizer=tokenizer,
        model=args.model,
        messages=example_messages_history,
        tools=available_tools
    )

    # Parse function calls and serialize formatted output
    parsed_calls = parse_tool_for_minicpm3(generate_result)
    formatted_data = format_conversation_history(
        messages=example_messages_history,
        generate_result=generate_result,
        new_tool_calls=parsed_calls
    )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    example_messages_history = [
        {
            "role": "system",
            "content": "You are an intelligent assistant with access to various tools."
        },
        {
            "role": "user",
            "content": "I'm searching for movie theaters in Hangzhou and wondering about the weather forecast for this evening."
        },
        {
            "role": "assistant",
            "content": "<|tool_call_start|>\n```python\nsearchPOI(city=\"杭州\",extensions=\"base\",keywords=\"电影院\")\n```\n<|tool_call_end|>\nI'll help you find movie theaters in Hangzhou."
        },
        {
            "role": "tool",
            "content": "{\"status\":\"1\",\"count\":229,\"pois\":[]}"
        }
    ]
    main()
