# Function Calling and Tool Use Logic for AVA
import json

class FunctionCaller:
    def __init__(self, available_tools):
        """Initializes the Function Caller.

        Args:
            available_tools (dict): A dictionary where keys are tool names and 
                                      values are callable functions or objects with a 'run' method.
        """
        self.available_tools = available_tools
        print(f"Function Caller initialized with tools: {list(self.available_tools.keys())}")

    def needs_function_call(self, llm_output):
        """Determines if the LLM output indicates a need for a function call.
        This could be based on specific keywords, structured output from the LLM,
        or a separate classification model.

        Args:
            llm_output (str or dict): The raw output from the LLM.

        Returns:
            bool: True if a function call is indicated, False otherwise.
        """
        # Example: Simple check for a specific string pattern
        # In a real system, this would be more robust, potentially parsing
        # structured JSON output if the LLM is trained for function calling.
        if isinstance(llm_output, str):
            return "[TOOL_CALL:" in llm_output
        elif isinstance(llm_output, dict) and "tool_calls" in llm_output:
            return bool(llm_output["tool_calls"])
        return False

    def parse_function_call(self, llm_output):
        """Parses the tool name and arguments from the LLM output.

        Args:
            llm_output (str or dict): The raw output from the LLM.

        Returns:
            tuple: (tool_name, tool_args_dict) or (None, None) if parsing fails.
        """
        # This needs to be robust and align with how the LLM is prompted/fine-tuned
        # to specify function calls.
        print(f"FC: Parsing LLM output for function call: {llm_output[:100]}...")
        if isinstance(llm_output, str) and "[TOOL_CALL:" in llm_output:
            try:
                # Example: [TOOL_CALL: get_weather(location="London", unit="celsius")]
                call_str = llm_output.split("[TOOL_CALL:")[1].split(")]")[0]
                tool_name = call_str.split("(")[0]
                args_str = call_str.split("(", 1)[1][:-1] # Get content within parentheses
                
                tool_args = {}
                if args_str: # Handle cases with no arguments
                    # This is a simplified parser for key="value" pairs
                    # A more robust solution might use ast.literal_eval or a small grammar
                    # or expect the LLM to output JSON directly for arguments.
                    for arg_pair in args_str.split(","):
                        key, value = arg_pair.split("=", 1)
                        # Attempt to parse value as JSON, fallback to string
                        try:
                            tool_args[key.strip()] = json.loads(value.strip())
                        except json.JSONDecodeError:
                            tool_args[key.strip()] = value.strip().strip('"').strip("'")
                
                if tool_name in self.available_tools:
                    print(f"FC: Parsed tool: {tool_name}, args: {tool_args}")
                    return tool_name, tool_args
                else:
                    print(f"FC: Warning - Parsed tool '{tool_name}' not in available tools.")
                    return None, None
            except Exception as e:
                print(f"FC: Error parsing function call string: {e}")
                return None, None
        # Placeholder for parsing structured JSON if LLM supports native function calling
        elif isinstance(llm_output, dict) and llm_output.get("tool_calls"):
            # Assuming the first tool call if multiple are present (for simplicity)
            # A real system would handle multiple parallel function calls if designed for it.
            first_tool_call = llm_output["tool_calls"][0]
            tool_name = first_tool_call.get("function", {}).get("name")
            try:
                tool_args_json = first_tool_call.get("function", {}).get("arguments")
                tool_args = json.loads(tool_args_json) if tool_args_json else {}
                if tool_name in self.available_tools:
                    print(f"FC: Parsed tool (JSON): {tool_name}, args: {tool_args}")
                    return tool_name, tool_args
                else:
                    print(f"FC: Warning - Parsed tool '{tool_name}' (JSON) not in available tools.")
                    return None, None
            except json.JSONDecodeError as e:
                print(f"FC: Error parsing JSON arguments for tool '{tool_name}': {e}")
                return None, None
            except Exception as e:
                print(f"FC: Error parsing structured function call: {e}")
                return None, None

        return None, None

    def execute_function(self, tool_name, tool_args):
        """Executes the specified tool with the given arguments.

        Args:
            tool_name (str): The name of the tool to execute.
            tool_args (dict): A dictionary of arguments for the tool.

        Returns:
            Any: The result of the tool execution, or an error message.
        """
        if tool_name in self.available_tools:
            tool_function = self.available_tools[tool_name]
            try:
                print(f"FC: Executing tool '{tool_name}' with args: {tool_args}")
                # If tool_args is None (no args), pass an empty dict or handle in tool
                result = tool_function(**(tool_args if tool_args is not None else {}))
                print(f"FC: Tool '{tool_name}' executed successfully. Result: {str(result)[:100]}...")
                return result
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(f"FC: {error_message}")
                return error_message # Return error message as result
        else:
            error_message = f"Tool '{tool_name}' not found."
            print(f"FC: {error_message}")
            return error_message

# --- Example Tools (for testing) ---
def get_weather(location, unit="celsius"):
    """Simulates fetching weather for a given location."""
    print(f"TOOL: get_weather called for {location} in {unit}")
    if location.lower() == "london":
        return {"temperature": "15", "condition": "Cloudy", "unit": unit}
    elif location.lower() == "tokyo":
        return {"temperature": "28", "condition": "Sunny", "unit": unit}
    else:
        return {"error": "Location not found"}

class Calculator:
    def run(self, operation, num1, num2):
        """Simulates a calculator tool."""
        print(f"TOOL: Calculator called for {operation} on {num1} and {num2}")
        operation = operation.lower()
        try:
            n1 = float(num1)
            n2 = float(num2)
            if operation == "add":
                return {"result": n1 + n2}
            elif operation == "subtract":
                return {"result": n1 - n2}
            elif operation == "multiply":
                return {"result": n1 * n2}
            elif operation == "divide":
                if n2 == 0:
                    return {"error": "Cannot divide by zero"}
                return {"result": n1 / n2}
            else:
                return {"error": f"Unknown operation: {operation}"}
        except ValueError:
            return {"error": "Invalid number input"}

if __name__ == "__main__":
    print("--- Function Caller Test ---")
    mock_tools = {
        "get_weather": get_weather,
        "calculator": Calculator().run # If it's a class method
    }
    fc = FunctionCaller(available_tools=mock_tools)

    # Test case 1: Simple string parsing
    llm_output_str = "Okay, I will use a tool. [TOOL_CALL: get_weather(location=\"Tokyo\", unit=\"celsius\")] Let me check that."
    print(f"\nTesting string output: {llm_output_str}")
    if fc.needs_function_call(llm_output_str):
        name, args = fc.parse_function_call(llm_output_str)
        if name:
            result = fc.execute_function(name, args)
            print(f"Result from {name}: {result}")

    # Test case 2: String parsing with no args
    llm_output_str_no_args = "[TOOL_CALL: special_action()]"
    # Add special_action to mock_tools for this test
    def special_action(): return "Special action performed!"
    fc.available_tools["special_action"] = special_action
    print(f"\nTesting string output (no args): {llm_output_str_no_args}")
    if fc.needs_function_call(llm_output_str_no_args):
        name, args = fc.parse_function_call(llm_output_str_no_args)
        if name:
            result = fc.execute_function(name, args)
            print(f"Result from {name}: {result}")
    del fc.available_tools["special_action"] # Clean up

    # Test case 3: Structured JSON output (simulated)
    llm_output_json = {
        "text_response": "I can help with that calculation.",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "calculator",
                    "arguments": json.dumps({"operation": "add", "num1": 5, "num2": 7})
                }
            }
        ]
    }
    print(f"\nTesting JSON output: {llm_output_json}")
    if fc.needs_function_call(llm_output_json):
        name, args = fc.parse_function_call(llm_output_json)
        if name:
            result = fc.execute_function(name, args)
            print(f"Result from {name}: {result}")

    # Test case 4: Tool not found
    llm_output_unknown_tool = "[TOOL_CALL: unknown_tool(param1=\"value\")]"
    print(f"\nTesting unknown tool: {llm_output_unknown_tool}")
    if fc.needs_function_call(llm_output_unknown_tool):
        name, args = fc.parse_function_call(llm_output_unknown_tool)
        if name: # Name might be parsed even if tool doesn't exist
            result = fc.execute_function(name, args)
            print(f"Result from {name}: {result}")
        else: # Or parsing itself might fail if structure is tied to known tools
            print("Could not parse unknown tool call properly.")

    # Test case 5: Error during tool execution
    llm_output_calc_error = {
        "tool_calls": [{"type": "function", "function": {"name": "calculator", "arguments": json.dumps({"operation": "divide", "num1": 5, "num2": 0})}}]
    }
    print(f"\nTesting tool execution error: {llm_output_calc_error}")
    if fc.needs_function_call(llm_output_calc_error):
        name, args = fc.parse_function_call(llm_output_calc_error)
        if name:
            result = fc.execute_function(name, args)
            print(f"Result from {name} (error expected): {result}")
    print("---------------------------")

print("Placeholder for AVA function calling logic (src/ava_core/function_calling.py)") 