# Main CLI Interface for AVA
import typer
from typing_extensions import Annotated

app = typer.Typer()

@app.command()
def query(prompt: Annotated[str, typer.Argument(help="The natural language query for AVA.")],
          model: Annotated[str, typer.Option(help="Specify the Ollama model to use (e.g., ava-agent:latest).")] = "ava-agent:latest",
          stream: Annotated[bool, typer.Option(help="Stream the response token by token.")] = True):
    """Sends a query to AVA and prints the response."""
    typer.echo(f"Querying AVA (model: {model}) with: '{prompt}' (Streaming: {stream})")
    # TODO: Implement actual call to Ollama or AVA's core logic
    # Example of how to integrate with Ollama python client:
    # try:
    #     import ollama
    #     response = ollama.chat(
    #         model=model,
    #         messages=[{'role': 'user', 'content': prompt}],
    #         stream=stream
    #     )
    #     if stream:
    #         for chunk in response:
    #             typer.echo(chunk['message']['content'], nl=False)
    #         typer.echo("") # Ensure a newline at the end
    #     else:
    #         typer.echo(response['message']['content'])
    # except ImportError:
    #     typer.secho("Ollama library not found. Please install it: pip install ollama", fg=typer.colors.RED)
    # except Exception as e:
    #     typer.secho(f"Error communicating with Ollama: {e}", fg=typer.colors.RED)
    
    if stream:
        simulated_response = ["This ", "is ", "a ", "simulated ", "streamed ", "response."]
        for chunk in simulated_response:
            typer.echo(chunk, nl=False)
        typer.echo("")
    else:
        typer.echo("This is a simulated non-streamed response.")

@app.command()
def chat(model: Annotated[str, typer.Option(help="Specify the Ollama model to use (e.g., ava-agent:latest).")] = "ava-agent:latest"):
    """Starts an interactive chat session with AVA."""
    typer.echo(f"Starting interactive chat with AVA (model: {model}). Type 'exit' or 'quit' to end.")
    # TODO: Implement persistent chat session logic
    #       - Maintain history
    #       - Call Ollama or AVA's core logic in a loop
    while True:
        prompt = typer.prompt("You")
        if prompt.lower() in ["exit", "quit"]:
            typer.echo("Exiting chat.")
            break
        typer.echo(f"AVA: You said '{prompt}'. (Simulated response)") # Replace with actual call

@app.command()
def status():
    """Checks the status of AVA and its connection to services like Ollama."""
    typer.echo("Checking AVA status...")
    # TODO: Implement checks for Ollama connection, model availability etc.
    # try:
    #     import ollama
    #     ollama.list()
    #     typer.secho("Ollama connection: OK", fg=typer.colors.GREEN)
    # except Exception as e:
    #     typer.secho(f"Ollama connection: FAILED ({e})", fg=typer.colors.RED)
    typer.echo("Ollama connection: (Simulated OK)")
    typer.echo("AVA Model (ava-agent:latest): (Simulated Loaded)")

@app.command()
def tool_exec(tool_name: Annotated[str, typer.Argument(help="The name of the tool to execute.")],
              args: Annotated[list[str], typer.Argument(help="Arguments for the tool in key=value format.")] = []):
    """Executes a specified tool with given arguments (for testing/development)."""
    parsed_args = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            parsed_args[key] = value
        else:
            typer.secho(f"Warning: Argument '{arg}' is not in key=value format and will be ignored.", fg=typer.colors.YELLOW)

    typer.echo(f"Executing tool: '{tool_name}' with arguments: {parsed_args}")
    # TODO: Integrate with AVA's FunctionCaller to actually execute the tool
    # from ava_core.function_calling import FunctionCaller
    # from src.tools.calculator import Calculator # Example tool
    # available_tools = {"calculator": Calculator()} 
    # caller = FunctionCaller(available_tools=available_tools)
    # result = caller.execute_tool(tool_name, parsed_args)
    # typer.echo(f"Tool Result: {result}")
    typer.echo(f"Simulated tool '{tool_name}' execution successful.")

if __name__ == "__main__":
    app()

print("Placeholder for AVA CLI (src/cli/main.py)") 