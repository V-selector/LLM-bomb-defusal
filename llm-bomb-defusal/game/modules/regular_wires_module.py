import random
import string
from game.modules.module import Module, ActionResult


class RegularWiresModule(Module):
    def __init__(self):
        super().__init__()
        self.wire_colors = []
        # Generate a serial number with letters and at least one digit
        self.serial_number = self._generate_serial_number()
        self.generate_wires()
    
    def _generate_serial_number(self) -> str:
        """Generate a random serial number with letters and at least one digit."""
        # Generate 5 random letters
        letters = ''.join(random.choices(string.ascii_uppercase, k=5))
        # Generate at least one digit
        digit = str(random.randint(0, 9))
        # Insert the digit at a random position
        position = random.randint(0, len(letters))
        return letters[:position] + digit + letters[position:]
    
    def generate_wires(self):
        """Generate a random set of wires for the module."""
        colors = ["red", "blue", "yellow", "white", "black"]
        num_wires = random.randint(3, 6)
        self.wire_colors = [random.choice(colors) for _ in range(num_wires)]
    
    def instruction(self) -> str:
        """Return the instruction manual for this module."""
        manual = """## Regular Wires Module

This module presents a series of colored wires (between 3 and 6). The defuser will see the colors and order of the wires,
as well as the bomb's serial number. You must determine which single wire should be cut based on the specific configuration
of wire colors and the serial number.

- 3-Wire Case:
  1. If no wires are red: Cut the second wire.
  2. Otherwise, if the last wire is white: Cut the last wire.
  3. Otherwise: Cut the last wire.

- 4-Wire Case:
  1. If there is more than one red wire and the last digit of the serial number is odd: Cut the last red wire.
  2. Else, if the last wire is yellow and there are no red wires: Cut the first wire.
  3. Else, if there is exactly one blue wire: Cut the first wire.
  4. Else, if there is more than one yellow wire: Cut the last wire.
  5. Otherwise: Cut the second wire.

- 5-Wire Case:
  1. If the last wire is black and the last digit of the serial number is odd: Cut the fourth wire.
  2. Else, if there is exactly one red wire and more than one yellow wire: Cut the first wire.
  3. Else, if there are no black wires: Cut the second wire.
  4. Otherwise: Cut the first wire.

- 6-Wire Case:
  1. If there are no yellow wires and the last digit of the serial number is odd: Cut the third wire.
  2. Else, if there is exactly one yellow wire and more than one white wire: Cut the fourth wire.
  3. Else, if there are no red wires: Cut the last wire.
  4. Otherwise: Cut the fourth wire."""
        return manual
    
    def _get_state(self) -> tuple[str, list[str]]:
        """Return the current state and available actions."""
        state_desc = f"Serial number: {self.serial_number}\n"
        state_desc += "Wires:\n"
        
        for i, color in enumerate(self.wire_colors, 1):
            state_desc += f"Wire {i}: {color}\n"
        
        actions = [f"cut wire {i+1}" for i in range(len(self.wire_colors))]
        return state_desc, actions
    
    def _do_action(self, action: str) -> ActionResult:
        """Perform the specified action."""
        try:
            # Extract wire number from action
            wire_num = int(action.lower().replace("cut wire ", "").strip())
            
            if wire_num < 1 or wire_num > len(self.wire_colors):
                return ActionResult.INCORRECT
            
            # Check if this is the correct wire to cut
            if self._is_correct_wire(wire_num):
                return ActionResult.DISARMED
            else:
                return ActionResult.EXPLODED
                
        except (ValueError, IndexError):
            return ActionResult.INCORRECT
    
    def _is_correct_wire(self, wire_num: int) -> bool:
        """Check if the wire is the correct one to cut based on the rules."""
        num_wires = len(self.wire_colors)
        
        # Convert to 0-based indexing
        wire_idx = wire_num - 1
        
        # Count occurrences of each color
        red_wires = self.wire_colors.count("red")
        blue_wires = self.wire_colors.count("blue")
        yellow_wires = self.wire_colors.count("yellow")
        white_wires = self.wire_colors.count("white")
        black_wires = self.wire_colors.count("black")
        
        # Find the last red wire (0-based index)
        last_red_idx = -1
        for i in range(len(self.wire_colors) - 1, -1, -1):
            if self.wire_colors[i] == "red":
                last_red_idx = i
                break
        
        # Get the last digit of the serial number
        last_digit = int([char for char in self.serial_number if char.isdigit()][-1])
        
        # Check if serial number is odd
        serial_odd = last_digit % 2 == 1
        
        # Apply rules based on number of wires
        if num_wires == 3:
            # If no red wires, cut the second wire
            if red_wires == 0:
                return wire_idx == 1
            # If the last wire is white, cut the last wire
            elif self.wire_colors[-1] == "white":
                return wire_idx == num_wires - 1
            # Otherwise, cut the last wire
            else:
                return wire_idx == num_wires - 1
                
        elif num_wires == 4:
            # If more than one red wire and serial number is odd, cut the last red wire
            if red_wires > 1 and serial_odd:
                return wire_idx == last_red_idx
            # If last wire is yellow and no red wires, cut the first wire
            elif self.wire_colors[-1] == "yellow" and red_wires == 0:
                return wire_idx == 0
            # If exactly one blue wire, cut the first wire
            elif blue_wires == 1:
                return wire_idx == 0
            # If more than one yellow wire, cut the last wire
            elif yellow_wires > 1:
                return wire_idx == num_wires - 1
            # Otherwise, cut the second wire
            else:
                return wire_idx == 1
                
        elif num_wires == 5:
            # If last wire is black and serial number is odd, cut the fourth wire
            if self.wire_colors[-1] == "black" and serial_odd:
                return wire_idx == 3
            # If exactly one red wire and more than one yellow wire, cut the first wire
            elif red_wires == 1 and yellow_wires > 1:
                return wire_idx == 0
            # If no black wires, cut the second wire
            elif black_wires == 0:
                return wire_idx == 1
            # Otherwise, cut the first wire
            else:
                return wire_idx == 0
                
        elif num_wires == 6:
            # If no yellow wires and serial number is odd, cut the third wire
            if yellow_wires == 0 and serial_odd:
                return wire_idx == 2
            # If exactly one yellow wire and more than one white wire, cut the fourth wire
            elif yellow_wires == 1 and white_wires > 1:
                return wire_idx == 3
            # If no red wires, cut the last wire
            elif red_wires == 0:
                return wire_idx == num_wires - 1
            # Otherwise, cut the fourth wire
            else:
                return wire_idx == 3
                
        return False
