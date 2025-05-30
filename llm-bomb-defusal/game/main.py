import sys
import random
from game.modules.module import ActionResult
from game.modules.regular_wires_module import RegularWiresModule
from game.modules.button_module import ButtonModule
from game.modules.memory_module import MemoryModule
from game.modules.simon_says_module import SimonSaysModule

# ANSI color codes
GREEN = "\033[92m"
WHITE = "\033[97m"
RESET = "\033[0m"


def print_separator():
    """Print a separator line."""
    print("=" * 50)


def print_help():
    """Print help information."""
    print("Keep Talking and Nobody Explodes - Text Edition")
    print("\nGame Description:")
    print("  In this game, there are two players:")
    print("  • The Defuser: Sees the bomb's current module 'screen' but not the manual.")
    print("  • The Manual Expert: Has access to the defusal manual but not the bomb's display.")
    print("  Players must collaborate by exchanging text instructions to solve each module.")
    print("  The goal is to disarm all modules before time runs out or the bomb explodes.")
    print("  Each module has specific rules that must be followed precisely.")
    print("  Communication is key - the defuser must clearly describe what they see,")
    print("  and the expert must provide clear instructions based on the manual.")
    print("\nUsage: python main.py [module_name]")
    print("\nAvailable modules:")
    print("  wires    - Regular Wires Module")
    print("  button   - The Button Module")
    print("  simon    - Simon Module")
    print("  memory   - Memory Module")
    print("  random   - Select a random module")
    print("\nIn-game commands:")
    print("  help     - Show this help")
    print("  manual   - Show the manual (for the expert)")
    print("  state    - Show the current state (for the defuser)")
    print("  quit     - Exit the game")


def get_module(module_name):
    """Get the appropriate module based on the name."""
    if module_name == "wires":
        return RegularWiresModule()
    elif module_name == "button":
        return ButtonModule()
    elif module_name == "simon":
        return SimonSaysModule()
    elif module_name == "memory":
        return MemoryModule()
    elif module_name == "random":
        return random.choice([
            RegularWiresModule(),
            ButtonModule(),
            SimonSaysModule(),
            MemoryModule()
        ])
    else:
        print(f"Unknown module: {module_name}")
        print_help()
        sys.exit(1)


def main():
    """Main function to run the game."""
    if len(sys.argv) != 2 or sys.argv[1] in ["-h", "--help"]:
        print_help()
        sys.exit(0)

    module_name = sys.argv[1].lower()
    module = get_module(module_name)

    print("Keep Talking and Nobody Explodes - Text Edition")
    print("\nType 'help' for commands, 'manual' for the expert, 'state' for the defuser.")

    while True:
        command = input(f"\n{GREEN}> ").strip().lower()
        print(f"{GREEN}Selected command: {command}{RESET}")

        if command == "quit" or command == "exit":
            print("Goodbye!")
            break

        elif command == "help":
            print_help()

        elif command == "manual":
            print("=" * 50)
            print("=== MANUAL (FOR THE EXPERT) ===")
            print("=" * 50)
            print(module.instruction())
            print("=" * 50)

        elif command == "state":
            print("=" * 50)
            print("=== BOMB STATE (FOR THE DEFUSER) ===")
            print("=" * 50)
            state, actions = module.state()
            print(state)
            if actions:
                print("\nAvailable actions:")
                for action in actions:
                    print(f"  {action}")
            print("=" * 50)

        elif command.startswith(("cut", "press", "hold", "release")):
            result = module.do_action(command)

            if result == ActionResult.CHANGED:
                print("The module state has changed.")
                # Show the updated state immediately after an action
                state, actions = module.state()
                print("\nCurrent state:")
                print(state)
                if actions:
                    print("\nAvailable actions:")
                    for action in actions:
                        print(f"  {action}")

            elif result == ActionResult.DISARMED:
                # module.set_disarmed()
                print("=" * 50)
                print("MODULE SUCCESSFULLY DISARMED! CONGRATULATIONS!")
                print("=" * 50)
                print("Type 'quit' to exit or start a new module.")

            elif result == ActionResult.EXPLODED:
                print("=" * 50)
                print("BOOM! THE BOMB HAS EXPLODED. GAME OVER.")
                print("=" * 50)
                print("Type 'quit' to exit or start a new module.")

            elif result == ActionResult.INCORRECT:
                print("That action is not valid or the module is already disarmed.")

        else:
            print("Unknown command. Type 'help' for available commands.")


if __name__ == "__main__":
    main()
