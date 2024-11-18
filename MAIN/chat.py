class Chat:
    def __init__(self):
        self.dialogues = {
            "introduction": "Hello, I'm AVA, your virtual assistant. How can I help you?",
            "goodbye": "Goodbye! Have a great day!",
            "unknown": "Sorry, I didn't understand that.",
        }

    def get_response(self, user_input):
        user_input = user_input.lower()
        if "hello" in user_input or "hi" in user_input:
            return self.dialogues["introduction"]
        elif "goodbye" in user_input or "bye" in user_input:
            return self.dialogues["goodbye"]
        else:
            return self.dialogues["unknown"]
