import keyboard

def on_key_press(event):
    print(f"Key code: {event.scan_code}, Key name: {event.name}")
    keyboard.unhook_all()  # Unhook the listener after detecting the key press

print("Press your play/pause button to see its key code and name.")
keyboard.on_press(on_key_press)
keyboard.wait('esc')  # Wait for the user to press the 'esc' key to exit
