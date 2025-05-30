from crewai.tools import BaseTool
import asyncio
from game_mcp.game_client import Defuser, Expert
from pydantic import BaseModel, Extra


SERVER_URL = 'http://localhost:8080'

class DefuserArgs(BaseModel):
    command: str
    class Config:
        extra = Extra.allow

class ExpertArgs(BaseModel):
    _input: str = ''
    class Config:
        extra = Extra.allow

class DefuserTool(BaseTool):
    """
    Tool that executes a defuser command on the bomb and returns the server response.
    """
    def __init__(self):
        super().__init__(
            name="defuser_action",
            description="Execute a defuser command on the bomb and return the server response.",
            args_schema=DefuserArgs
        )

    def _run(self, **kwargs) -> str:
        """
        Executes a defuser command and returns the server's response.
        """
        args = DefuserArgs(**kwargs)
        client = Defuser()
        async def run():
            await client.connect_to_server(SERVER_URL)
            res = await client.run(args.command)
            await client.cleanup()
            return res
        return asyncio.run(run())

class ExpertTool(BaseTool):
    """
    Tool that retrieves the current module's instructions from the manual.
    """
    def __init__(self):
        super().__init__(
            name="expert_manual",
            description="Retrieve the current module's instructions from the manual.",
            args_schema=ExpertArgs
        )

    def _run(self, **kwargs) -> str:
        """
        Retrieves the manual instructions for the current bomb module.
        """
        _ = ExpertArgs(**kwargs)
        client = Expert()
        async def run():
            await client.connect_to_server(SERVER_URL)
            res = await client.run()
            await client.cleanup()
            return res
        return asyncio.run(run())
