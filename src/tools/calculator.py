# Calculator Tool for AVA

import math

class Calculator:
    def __init__(self):
        self.name = "calculator"
        self.description = "Performs basic arithmetic calculations. Supports addition (+), subtraction (-), multiplication (*), division (/), exponentiation (^), and square root (sqrt)."
        self.parameters = [
            {"name": "expression", "type": "string", "description": "The mathematical expression to evaluate. Example: '2 + 2', 'sqrt(16)', '3^4'."}
        ]
        print("Calculator tool initialized.")

    def _is_safe_expression(self, expression):
        """Basic check to allow only safe characters and functions."""
        allowed_chars = "0123456789.+-*/^() "
        allowed_funcs = ["sqrt"]
        
        # Check for allowed characters
        for char in expression:
            if char not in allowed_chars and not char.isalpha(): # alpha for functions
                return False
        
        # Check for allowed functions (simple check, can be improved)
        temp_expr = expression
        for func in allowed_funcs:
            temp_expr = temp_expr.replace(func, "")
        
        # After removing known funcs, only allowed_chars should remain
        for char in temp_expr:
            if char not in allowed_chars:
                return False
        return True

    def run(self, expression_str):
        """Executes the calculation.

        Args:
            expression_str (str): The mathematical expression string.

        Returns:
            dict: A dictionary containing the result or an error message.
                  Example: {"result": 4} or {"error": "Invalid expression"}
        """
        print(f"Calculator tool: Received expression '{expression_str}'")
        
        if not self._is_safe_expression(expression_str):
            print("Calculator tool: Unsafe expression detected.")
            return {"error": "Invalid or unsafe characters/functions in expression."}

        try:
            # Replace ^ with ** for Python's power operator
            expression_to_eval = expression_str.replace('^', '**')
            
            # Define a limited scope for eval with math functions if needed
            safe_dict = {
                'sqrt': math.sqrt,
                # Add other math functions if you want to support them directly
                # Be very careful about what you expose to eval
            }
            
            # Using eval is generally risky, but here it's somewhat controlled by _is_safe_expression
            # and by providing a limited globals/locals dict.
            # For production, a dedicated expression parser would be much safer.
            result = eval(expression_to_eval, {"__builtins__": {}}, safe_dict)
            print(f"Calculator tool: Result = {result}")
            return {"result": result}
        except ZeroDivisionError:
            print("Calculator tool: Error - Division by zero.")
            return {"error": "Division by zero."}
        except Exception as e:
            print(f"Calculator tool: Error evaluating expression - {str(e)}")
            return {"error": f"Invalid expression or calculation error: {str(e)}"}

if __name__ == "__main__":
    calc = Calculator()
    print(f"\nTool: {calc.name}\nDescription: {calc.description}\nParams: {calc.parameters}")
    
    expressions_to_test = [
        "2 + 2",
        "10 - 5.5",
        "3 * 7",
        "10 / 2",
        "2^8",
        "sqrt(16)",
        "(2 + 3) * 4",
        "10 / 0", # Error case: division by zero
        "sqrt(-4)", # Error case: math domain error
        "import os" # Unsafe, should be caught
    ]
    
    for expr in expressions_to_test:
        print(f"\nTesting expression: '{expr}'")
        output = calc.run(expr)
        print(f"Output: {output}")

print("Placeholder for Calculator tool (src/tools/calculator.py)") 