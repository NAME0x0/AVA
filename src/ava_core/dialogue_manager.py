# Dialogue Manager for AVA

class DialogueManager:
    def __init__(self, max_history_length=20):
        """Initializes the Dialogue Manager.

        Args:
            max_history_length (int): Maximum number of turns to keep in history.
        """
        self.history = []
        self.max_history_length = max_history_length
        print("Dialogue Manager initialized.")

    def add_message(self, role, content):
        """Adds a message to the dialogue history.

        Args:
            role (str): 'user' or 'assistant'.
            content (str): The message content.
        """
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be 'user' or 'assistant'.")
        
        self.history.append({"role": role, "content": content})
        # Truncate history if it exceeds max length
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]
        print(f"DM: Added {role} message: \"{content[:50]}...\"")

    def get_context(self, include_system_prompt=True):
        """Constructs the current conversational context for the LLM.

        Args:
            include_system_prompt (bool): Whether to prepend a system prompt.

        Returns:
            str or list: Formatted context suitable for the LLM.
                         (Could be a list of dicts for chat models, or a single string)
        """
        context_to_send = []
        if include_system_prompt:
            # This system prompt should ideally be configurable
            context_to_send.append({
                "role": "system", 
                "content": "You are AVA, a helpful and highly capable AI assistant. Be concise. If you need to use a tool, use the format [TOOL_CALL: tool_name(param1=value1, param2=value2)]."
            })
        
        context_to_send.extend(self.history)
        print(f"DM: Context provided with {len(context_to_send)} items.")
        # For some models, you might return a string instead:
        # return "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in context_to_send])
        return context_to_send # Assuming a chat model format

    def get_context_with_tool_result(self, tool_name, tool_result):
        """Constructs context including the result of a tool call.
        This is often a special user message or a dedicated tool role message.
        """
        # Add a temporary message for the tool result to the history for the next LLM call
        # The exact format depends on how the LLM is fine-tuned to handle tool results.
        # Option 1: As a user message
        # tool_response_message = {"role": "user", "content": f"[TOOL_RESULT tool_name='{tool_name}']: {tool_result}"}
        # Option 2: As a dedicated tool role (if supported by the model/fine-tuning)
        tool_response_message = {"role": "tool", "content": tool_result, "tool_call_id": tool_name} # Simplified tool_call_id
        
        current_context = self.get_context(include_system_prompt=True)
        current_context.append(tool_response_message)
        print(f"DM: Context provided with tool result for '{tool_name}'.")
        return current_context

    def clear_history(self):
        """Clears the dialogue history."""
        self.history = []
        print("DM: History cleared.")

if __name__ == "__main__":
    print("--- Dialogue Manager Test ---")
    dm = DialogueManager(max_history_length=5)
    dm.add_message("user", "Hello AVA!")
    dm.add_message("assistant", "Hello! How can I help you today?")
    dm.add_message("user", "What is the weather in London?")
    dm.add_message("assistant", "[TOOL_CALL: get_weather(location='London')]")
    
    context = dm.get_context()
    print(f"Current Context:\n{json.dumps(context, indent=2)}\n")

    context_with_tool_result = dm.get_context_with_tool_result("get_weather", "Sunny, 25 C")
    print(f"Context with Tool Result:\n{json.dumps(context_with_tool_result, indent=2)}\n")

    dm.add_message("user", "Thanks!") # This will cause truncation if max_history_length is 5
    dm.add_message("assistant", "You are welcome!")
    context_after_truncation = dm.get_context()
    print(f"Context after more messages (and potential truncation):\n{json.dumps(context_after_truncation, indent=2)}")
    
    dm.clear_history()
    print(f"History after clear: {dm.history}")
    print("---------------------------")

print("Placeholder for AVA dialogue manager (src/ava_core/dialogue_manager.py)")
import json # Added for testing print output 