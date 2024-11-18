import os
import time

# Function to speak text using Narrator app
def speak_text(text):
    # Use os.system to invoke the Narrator app and read the provided text
    os.system(f'Narrator.exe /s "{text}"')

# Function to clear focus from the terminal window
def clear_focus():
    # Create and focus on a temporary blank window
    os.system("start cmd /c echo. && pause >nul")

# Function to close the Narrator app
def close_narrator():
    # Clear focus to prevent Narrator from reading unwanted labels
    clear_focus()
    # Terminate the Narrator process
    os.system("taskkill /im Narrator.exe /f")

def main():
    print("Testing Windows Narrator Text-to-Speech")

    tongue_twisters = [
        "How can a clam cram in a clean cream can?",
        "She sells seashells by the seashore.",
        "Peter Piper picked a peck of pickled peppers."
    ]

    math_operation = "Eight plus five equals thirteen."

    # Launch the Narrator app at the beginning
    os.system("start Narrator.exe")

    print("\nTongue Twisters:")
    for i, twister in enumerate(tongue_twisters, start=1):
        print(f"Tongue Twister {i}:")
        print(twister)
        speak_text(twister)
        time.sleep(2)  # Add a short delay between speaking each tongue twister

    print("\nMathematical Operation:")
    print(math_operation)
    speak_text(math_operation)

    # Close the Narrator app at the end
    close_narrator()

if __name__ == "__main__":
    main()
