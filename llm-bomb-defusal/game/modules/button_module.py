import random
from game.modules.module import Module, ActionResult


class ButtonModule(Module):
    def __init__(self):
        super().__init__()
        self.colors = ["red", "blue", "white", "yellow"]
        self.labels = ["Abort", "Detonate", "Hold", "Press"]
        
        self.button_color = random.choice(self.colors)
        self.button_label = random.choice(self.labels)
        
        self.batteries = random.randint(0, 4)
        self.lit_indicators = []
        
        # Randomly decide if CAR and FRK indicators are present and lit
        if random.choice([True, False]):
            self.lit_indicators.append("CAR")
        if random.choice([True, False]):
            self.lit_indicators.append("FRK")
            
        self.is_holding = False
        self.strip_color = None
    
    def instruction(self) -> str:
        """Return the instruction manual for this module."""
        manual = """## The Button Module

This module presents a single colored button with a text label. The defuser will see the button's color, label, 
number of batteries, and any lit indicators on the bomb. Based on these attributes, you must determine whether 
the defuser should press and immediately release the button, or hold it and release at a specific time.

1. Primary Analysis (Button Color and Label):
   - If the button is blue and labeled "Abort": Hold the button.
   - If there is more than one battery and the button says "Detonate": Press and immediately release the button.
   - If the button is white and there is a lit indicator labeled CAR: Hold the button.
   - If there are more than two batteries and a lit indicator labeled FRK is present: Press and immediately release the button.
   - If the button is yellow: Hold the button.
   - If the button is red and the button says "Hold": Press and immediately release the button.
   - Otherwise: Hold the button.

2. Releasing a Held Button (if required):
   When a held button produces a colored strip on its side, release when:
   - Blue strip: Release when the countdown timer shows any 4.
   - White strip: Release when the countdown timer shows any 1.
   - Yellow strip: Release when the countdown timer shows any 5.
   - Any other color: Release when the timer shows any 1."""
        return manual
    
    def _get_state(self) -> tuple[str, list[str]]:
        """Return the current state and available actions."""
        state_desc = f"Button: {self.button_color} button labeled '{self.button_label}'\n"
        state_desc += f"Batteries: {self.batteries}\n"
        
        if self.lit_indicators:
            state_desc += f"Lit indicators: {', '.join(self.lit_indicators)}\n"
        else:
            state_desc += "No lit indicators\n"
        
        if self.is_holding:
            state_desc += f"You are holding the button. A {self.strip_color} colored strip is visible.\n"
            actions = ["release on 1", "release on 4", "release on 5"]
        else:
            actions = ["press", "hold"]
        
        return state_desc, actions
    
    def _do_action(self, action: str) -> ActionResult:
        """Perform the specified action."""
        action = action.lower().strip()
        
        if not self.is_holding:
            if action == "press":
                if self._should_press():
                    return ActionResult.DISARMED
                else:
                    return ActionResult.EXPLODED
            elif action == "hold":
                self.is_holding = True
                self.strip_color = random.choice(["blue", "white", "yellow", "red", "green"])
                return ActionResult.CHANGED
            else:
                return ActionResult.INCORRECT
        else:  # Currently holding
            if action.startswith("release on "):
                try:
                    digit = int(action.replace("release on ", ""))
                    correct_digit = self._get_correct_release_digit()
                    
                    if digit == correct_digit:
                        return ActionResult.DISARMED
                    else:
                        return ActionResult.EXPLODED
                except ValueError:
                    return ActionResult.INCORRECT
            else:
                return ActionResult.INCORRECT
    
    def _should_press(self) -> bool:
        """Determine if the button should be pressed (not held) based on the rules."""
        # If there is more than one battery and the button says "Detonate"
        if self.batteries > 1 and self.button_label == "Detonate":
            return True
        
        # If there are more than two batteries and a lit indicator labeled FRK is present
        if self.batteries > 2 and "FRK" in self.lit_indicators:
            return True
        
        # If the button is red and the button says "Hold"
        if self.button_color == "red" and self.button_label == "Hold":
            return True
        
        # In all other cases, the button should be held, not pressed
        return False
    
    def _get_correct_release_digit(self) -> int:
        """Get the correct digit for releasing the button based on strip color."""
        if self.strip_color == "blue":
            return 4
        elif self.strip_color == "yellow":
            return 5
        else:  # white or any other color
            return 1
