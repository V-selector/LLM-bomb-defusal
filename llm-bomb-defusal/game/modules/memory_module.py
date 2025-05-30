import random
from game.modules.module import Module, ActionResult


class MemoryModule(Module):
    def __init__(self):
        super().__init__()
        self.current_stage = 1
        self.max_stages = 5
        self.display_number = 0
        self.button_labels = []
        self.stage_history = {}  # Stores position and label for each stage
        self.generate_stage()
    
    def generate_stage(self):
        """Generate a new stage with a display number and button labels."""
        self.display_number = random.randint(1, 4)
        # Generate 4 unique button labels (1-4)
        self.button_labels = random.sample(range(1, 5), 4)
    
    def instruction(self) -> str:
        """Return the instruction manual for this module."""
        manual = """## Memory Module

This module has a display showing a digit (1-4) and four buttons labeled 1-4 in different positions.
The module has 5 stages, and each stage requires pressing a specific button based on the display value
and the history of previous stages.

Stage 1:
- If the display shows 1: Press the button in position 2
- If the display shows 2: Press the button in position 2
- If the display shows 3: Press the button in position 3
- If the display shows 4: Press the button in position 4

Stage 2:
- If the display shows 1: Press the button labeled 4
- If the display shows 2: Press the button in the same position as stage 1
- If the display shows 3: Press the button in position 1
- If the display shows 4: Press the button in the same position as stage 1

Stage 3:
- If the display shows 1: Press the button with the same label as you pressed in stage 2
- If the display shows 2: Press the button with the same label as you pressed in stage 1
- If the display shows 3: Press the button in position 3
- If the display shows 4: Press the button labeled 4

Stage 4:
- If the display shows 1: Press the button in the same position as stage 1
- If the display shows 2: Press the button in position 1
- If the display shows 3: Press the button in the same position as stage 2
- If the display shows 4: Press the button in the same position as stage 2

Stage 5:
- If the display shows 1: Press the button with the same label as you pressed in stage 1
- If the display shows 2: Press the button with the same label as you pressed in stage 2
- If the display shows 3: Press the button with the same label as you pressed in stage 4
- If the display shows 4: Press the button with the same label as you pressed in stage 3

Note: "Position" refers to the physical location (1-4 from left to right), while "Label" refers to the number shown on the button."""
        return manual
    
    def _get_state(self) -> tuple[str, list[str]]:
        """Return the current state and available actions."""
        if self.is_disarmed:
            return "Module disarmed!", []

        state_desc = f"Stage {self.current_stage}/{self.max_stages}\n"
        state_desc += f"Display shows: {self.display_number}\n"
        state_desc += "Buttons (position: label):\n"
        
        for position, label in enumerate(self.button_labels, 1):
            state_desc += f"Position {position}: Button labeled {label}\n"
        
        actions = [f"press position {i}" for i in range(1, 5)]
        return state_desc, actions
    
    def _do_action(self, action: str) -> ActionResult:
        """Perform the specified action."""
        try:
            # Extract position from action
            position = int(action.lower().replace("press position ", "").strip())
            
            if position < 1 or position > 4:
                return ActionResult.INCORRECT
            
            # Get the label at the selected position (0-indexed in the list)
            selected_label = self.button_labels[position - 1]
            
            # Check if this is the correct position to press
            if self._is_correct_position(position):
                # Store the position and label for this stage
                self.stage_history[self.current_stage] = {
                    "position": position,
                    "label": selected_label
                }
                
                # Move to the next stage
                self.current_stage += 1
                
                # Check if all stages are completed
                if self.current_stage > self.max_stages:
                    self.set_disarmed()
                    return ActionResult.DISARMED
                
                # Generate the next stage
                self.generate_stage()
                return ActionResult.CHANGED
            else:
                return ActionResult.EXPLODED
                
        except (ValueError, IndexError):
            return ActionResult.INCORRECT
    
    def _is_correct_position(self, position: int) -> bool:
        """Check if the position is correct based on the current stage and display number."""
        # Stage 1 rules
        if self.current_stage == 1:
            if self.display_number == 1:
                return position == 2
            elif self.display_number == 2:
                return position == 2
            elif self.display_number == 3:
                return position == 3
            elif self.display_number == 4:
                return position == 4
        
        # Stage 2 rules
        elif self.current_stage == 2:
            if self.display_number == 1:
                # Press the button labeled 4
                return self.button_labels[position - 1] == 4
            elif self.display_number == 2:
                # Press the button in the same position as stage 1
                return position == self.stage_history[1]["position"]
            elif self.display_number == 3:
                # Press the button in position 1
                return position == 1
            elif self.display_number == 4:
                # Press the button in the same position as stage 1
                return position == self.stage_history[1]["position"]
        
        # Stage 3 rules
        elif self.current_stage == 3:
            if self.display_number == 1:
                # Press the button with the same label as stage 2
                return self.button_labels[position - 1] == self.stage_history[2]["label"]
            elif self.display_number == 2:
                # Press the button with the same label as stage 1
                return self.button_labels[position - 1] == self.stage_history[1]["label"]
            elif self.display_number == 3:
                # Press the button in position 3
                return position == 3
            elif self.display_number == 4:
                # Press the button labeled 4
                return self.button_labels[position - 1] == 4
        
        # Stage 4 rules
        elif self.current_stage == 4:
            if self.display_number == 1:
                # Press the button in the same position as stage 1
                return position == self.stage_history[1]["position"]
            elif self.display_number == 2:
                # Press the button in position 1
                return position == 1
            elif self.display_number == 3 or self.display_number == 4:
                # Press the button in the same position as stage 2
                return position == self.stage_history[2]["position"]
        
        # Stage 5 rules
        elif self.current_stage == 5:
            if self.display_number == 1:
                # Press the button with the same label as stage 1
                return self.button_labels[position - 1] == self.stage_history[1]["label"]
            elif self.display_number == 2:
                # Press the button with the same label as stage 2
                return self.button_labels[position - 1] == self.stage_history[2]["label"]
            elif self.display_number == 3:
                # Press the button with the same label as stage 4
                return self.button_labels[position - 1] == self.stage_history[4]["label"]
            elif self.display_number == 4:
                # Press the button with the same label as stage 3
                return self.button_labels[position - 1] == self.stage_history[3]["label"]
        
        return False
