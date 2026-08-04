import json
import ast
import operator
import requests

def safe_eval(expr: str):
    """Safely evaluate basic mathematical expressions without using eval()."""
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.Mod: operator.mod
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Fallback for older python versions
            return node.n
        elif isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported operation or expression structure.")

    return _eval(ast.parse(expr, mode='eval').body)

# Define tool schemas for local LLM function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_action_item",
            "description": "Create a task with assignee, deadline and priority",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task description"},
                    "assignee": {"type": "string", "description": "Person responsible"},
                    "deadline": {"type": "string", "description": "Due date"},
                    "priority": {"type": "string", "description": "high, medium, or low"}
                },
                "required": ["task", "assignee", "deadline", "priority"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a math calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"]
            }
        }
    }
]

meeting_notes = """
Team standup July 8: John said the API migration is behind schedule, needs to finish by Friday July 11. 
Maria will review the Q3 budget, we're over by 15 percent on a 50000 dollar budget so need to find cuts. 
Raj is handling the client demo on Monday July 14, high priority. 
Sarah needs to update the documentation by end of week, low priority.
"""

def process_meeting_notes():
    # Update endpoint to your specific running server (e.g., SGLang: 30000, vLLM: 8000, LMStudio: 1234)
    endpoint = "http://localhost:30000/v1/chat/completions"
    
    print("=== Meeting Notes → Action Items (MiniCPM5-1B) ===\n")
    print(f"Input:\n{meeting_notes.strip()}\n")
    print("Processing...\n")

    payload = {
        "model": "openbmb/MiniCPM5-1B", # Ensure this matches your server's registered model tag
        "messages": [
            {
                "role": "system", 
                "content": "You are a meeting assistant. Extract ALL action items from the notes. Use create_action_item for each task. Use calculator for any math needed."
            },
            {
                "role": "user", 
                "content": f"Extract action items from these meeting notes:\n{meeting_notes}"
            }
        ],
        "tools": tools,
        "temperature": 0.1 # Lower temperature heavily enforces strict tool calling rules
    }

    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        msg = data["choices"][0]["message"]

        if msg.get("tool_calls"):
            print(f"Found {len(msg['tool_calls'])} tool call(s):\n")
            for i, tc in enumerate(msg["tool_calls"], 1):
                raw_args = tc["function"]["arguments"]
                
                # Resilient JSON parsing guard
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    print(f" [{i}] Error: Failed to parse arguments string: {raw_args}\n")
                    continue

                func = tc["function"]["name"]

                if func == "create_action_item":
                    print(f" [{i}] Task: {args.get('task', 'N/A')}")
                    print(f"     Assignee: {args.get('assignee', 'N/A')}")
                    print(f"     Deadline: {args.get('deadline', 'N/A')}")
                    print(f"     Priority: {args.get('priority', 'N/A')}\n")
                elif func == "calculator":
                    expr = args.get("expression", "")
                    try:
                        result = safe_eval(expr)
                        print(f" [CALC] Expression: {expr} = {result}\n")
                    except Exception as err:
                        print(f" [CALC Error] Failed to evaluate '{expr}': {err}\n")
        else:
            print(f"Response: {msg.get('content', 'No content or tool calls returned.')}")

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the local endpoint at {endpoint}.")
        print("Please ensure your local server backend is actively running.")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"An error occurred: {err}")

if __name__ == "__main__":
    process_meeting_notes()
