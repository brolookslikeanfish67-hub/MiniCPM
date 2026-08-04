import argparse
import ast
import asyncio
import json
import logging
import operator
from typing import Any, Dict, List, Optional, Union
import httpx

# Configure professional structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("MiniCPMAgent")

class SafeMathEngine:
    """A completely sandbox-isolated mathematical execution environment."""
    
    _ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow
    }

    @classmethod
    def evaluate(cls, expression: str) -> Union[int, float, str]:
        """Parses and computes mathematical strings safely without using eval()."""
        if not expression or not isinstance(expression, str):
            return "Error: Invalid expression type"
            
        try:
            tree = ast.parse(expression.strip(), mode='eval')
            return cls._secure_ast_eval(tree.body)
        except KeyError as e:
            return f"Security Error: Unsupported mathematical operator {str(e)}"
        except Exception as e:
            return f"Calculation Error: {str(e)}"

    @classmethod
    def _secure_ast_eval(cls, node: Any) -> Union[int, float]:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Backward compatibility fallback
            return node.n
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            return cls._ALLOWED_OPS[op_type](cls._secure_ast_eval(node.left), cls._secure_ast_eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            return cls._ALLOWED_OPS[op_type](cls._secure_ast_eval(node.operand))
        raise ValueError(f"Unauthorized AST Node structure detected: {type(node).__name__}")


class MiniCPMAgentEngine:
    """Advanced runtime engine orchestrating async tool loops for MiniCPM5-1B."""

    def __init__(self, endpoint: str, model_name: str, provider: str = "sglang"):
        self.endpoint = endpoint.rstrip('/')
        self.model_name = model_name
        self.provider = provider.lower()
        self.tools = self._get_tool_registry()

    @staticmethod
    def _get_tool_registry() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_action_item",
                    "description": "Log an extracted meeting task with full core context metadata.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "Granular task description"},
                            "assignee": {"type": "string", "description": "Full name of owner"},
                            "deadline": {"type": "string", "description": "Target absolute or relative due date"},
                            "priority": {"type": "string", "enum": ["high", "medium", "low"], "description": "Task urgency tier"}
                        },
                        "required": ["task", "assignee", "deadline", "priority"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Solve math formulas, percentages, or conversions safely.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "The math equation to solve"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]

    def _build_payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Constructs backend-specific hyper-parameter configurations dynamically."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": self.tools,
            "tool_choice": "auto",
            "temperature": 0.0,  
            "max_tokens": 1024
        }
        
        if self.provider in ["sglang", "vllm"]:
            payload["extra_body"] = {"enable_thinking": False}  
            
        return payload

    async def execute_workflow_async(self, input_text: str) -> Optional[str]:
        """Main ReAct asynchronous wrapper. Updates payload, loops keys, returns text summary."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an elite Operations Assistant. Extract EVERY single task from the input notes. "
                    "Process computations using 'calculator'. Structuralize tasks via 'create_action_item'. "
                    "After tools execute, summarize the operational plan."
                )
            },
            {"role": "user", "content": f"Analyze context and map workflows:\n\n{input_text}"}
        ]

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                logger.info(f"Initiating agent step-1 loop on target inference framework: {self.provider.upper()}")
                
                response = await client.post(self.endpoint, json=self._build_payload(messages))
                response.raise_for_status()
                
                # FIXED: Correctly access the JSON object key array hierarchy
                msg_node = response.json()["choices"][0]["message"]
                tool_calls = msg_node.get("tool_calls")

                if not tool_calls:
                    logger.warning("Agent fallback: Model skipped tool calls on step 1.")
                    return msg_node.get("content")

                messages.append(msg_node)
                logger.info(f"Intercepted {len(tool_calls)} system function call triggers.")

                for tool in tool_calls:
                    func_name = tool["function"]["name"]
                    raw_args = tool["function"]["arguments"]
                    
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    logger.info(f"Processing execution module [{func_name}]")

                    if func_name == "calculator":
                        expr = args.get("expression", "")
                        calc_result = SafeMathEngine.evaluate(expr)
                        tool_output = json.dumps({"result": str(calc_result)})
                        print(f" 🧮 [MATH ENGINE EXECUTED] Formula: {expr} ➔ Result: {calc_result}")

                    elif func_name == "create_action_item":
                        tool_output = json.dumps({"status": "logged_in_database"})
                        print(f" 📋 [TASK STRUCT LOGGED]")
                        print(f"    ├─ Owner:    {args.get('assignee')}\n    ├─ Workflow: {args.get('task')}")
                        print(f"    └─ Priority: [{args.get('priority').upper()}] | Due: {args.get('deadline')}\n")
                    else:
                        tool_output = json.dumps({"error": "Unknown system endpoint function connection"})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool.get("id", "call_untracked"),
                        "name": func_name,
                        "content": tool_output
                    })

                logger.info("Re-routing loop with execution logs to construct definitive overview summary...")
                final_response = await client.post(self.endpoint, json=self._build_payload(messages))
                final_response.raise_for_status()
                
                # FIXED: Correctly access the final conversational layer array
                final_msg = final_response.json()["choices"][0]["message"]
                return final_msg.get("content")

            except httpx.ConnectError:
                logger.error(f"Inference link connection failure at [{self.endpoint}]. Is your service wrapper open?")
                return None
            except Exception as ex:
                logger.error(f"Fatal anomaly intercepted inside execution loop sequence: {str(ex)}")
                return None


PROD_MEETING_NOTES = """
Team standup July 8: John said the API migration is behind schedule, needs to finish by Friday July 11. 
Maria will review the Q3 budget, we're over by 15 percent on a 50000 dollar budget so need to find cuts. 
Raj is handling the client demo on Monday July 14, high priority. 
Sarah needs to update the documentation by end of week, low priority.
"""

def main():
    parser = argparse.ArgumentParser(description="Advanced Production-Ready MiniCPM5 Agent Framework")
    parser.add_argument("--url", type=str, default="http://localhost:30000/v1/chat/completions", help="Complete server endpoint URI.")
    parser.add_argument("--model", type=str, default="openbmb/MiniCPM5-1B", help="Registered framework model signature.")
    parser.add_argument("--provider", type=str, choices=["sglang", "vllm", "lmstudio"], default="sglang", help="Underlying server infrastructure.")
    args = parser.parse_args()

    agent = MiniCPMAgentEngine(endpoint=args.url, model_name=args.model, provider=args.provider)
    
    print("\n" + "="*60)
    print("      MINICPM5-1B ADVANCED AGENT REAL-TIME WORKFLOW")
    print("="*60 + "\n")
    
    final_analysis = asyncio.run(agent.execute_workflow_async(PROD_MEETING_NOTES))
    
    if final_analysis:
        print("\n" + "="*60)
        print("      AGENT FINAL SYNTHESIZED SYSTEM OPERATIONS REPORT")
        print("="*60)
        print(final_analysis)
print("="*60 + "\n")if name == "main":main()
