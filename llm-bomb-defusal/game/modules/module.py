from enum import Enum


class ActionResult(Enum):
    CHANGED = "Changed"
    DISARMED = "Disarmed"
    EXPLODED = "Exploded"
    INCORRECT = "Incorrect"


class Module:
    def __init__(self):
        self.is_disarmed = False
    
    def set_disarmed(self):
        """Set the module as disarmed."""
        self.is_disarmed = True
    
    def instruction(self) -> str:
        """
        Returns the instruction for the manual expert.
        Only accessible to the player with manual.
        """
        raise NotImplementedError("Subclasses must implement instruction()")
    
    def state(self) -> tuple[str, list[str]]:
        """
        Returns the current state and available actions.
        Only accessible to the defuser.
        
        Returns:
            tuple: (state description, list of available actions)
        """
        if self.is_disarmed:
            return "Module disarmed!", []
        return self._get_state()
    
    def _get_state(self) -> tuple[str, list[str]]:
        """
        Returns the current state and available actions when the module is not disarmed.
        To be implemented by subclasses.
        """
        return "text", ["text"]
        
    
    def do_action(self, action: str) -> ActionResult:
        """
        Performs the specified action.
        Only accessible to the defuser.
        
        Args:
            action: The action to perform
            
        Returns:
            ActionResult: The result of the action
        """
        if self.is_disarmed:
            return ActionResult.INCORRECT
        return self._do_action(action)
    
    def _do_action(self, action: str) -> ActionResult:
        """
        Performs the specified action when the module is not disarmed.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _do_action()")
