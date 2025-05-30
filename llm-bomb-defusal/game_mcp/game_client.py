import asyncio
import argparse
import json
import ast
import urllib.parse

import aiohttp
from aiohttp_sse_client import client as sse_client


# Feel free to import any libraries you need - if needed change requirements.txt


class BombClient:
    def __init__(self):
        # YOUR CODE STARTS HERE
        self.session: aiohttp.ClientSession | None = None
        self.event_source: sse_client.EventSource | None = None
        self.session_url: str | None = None
        self._id_counter = 1
        # YOUR CODE ENDS HERE

    async def connect_to_server(self, server_url: str):
        """Connect to an sse MCP server"""
        # YOUR CODE STARTS HERE
        base = server_url.rstrip("/")
        self.session = aiohttp.ClientSession()
        self.event_source = sse_client.EventSource(f"{base}/")
        await self.event_source.connect()

        # Step 1: read until we get the session_id URL
        async for event in self.event_source:
            raw = event.data.strip().strip("'\"")
            if "/session_id/" in raw and "session_id=" in raw:
                parsed = urllib.parse.urlparse(raw)
                qs = urllib.parse.parse_qs(parsed.query)
                sid = qs.get("session_id", [None])[0]
                if sid:
                    self.session_url = f"{base}{parsed.path}?session_id={sid}"
                    break

        # Step 2: send initialize handshake (with clientInfo.version!)
        init_id = self._id_counter
        init_payload = {
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "1.0",
                "clientInfo": {
                    "name": type(self).__name__,
                    "version": "1.0"
                },
                "capabilities": {}
            },
        }
        self._id_counter += 1

        await self.session.post(self.session_url, json=init_payload)

        # Step 3: await initialize response and send notification
        async for event in self.event_source:
            raw = event.data.strip()
            # skip session_id lines
            if raw.startswith("/") or raw.startswith("'/"):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("id") == init_id:
                # once we see the initialize response, send the 'initialized' notification:
                notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                await self.session.post(self.session_url, json=notification)
                return
        # YOUR CODE ENDS HERE

    async def process_query(self, tool_name: str, tool_args: dict[str, str]) -> str:
        """Process a query using the game_interaction or get_manual tool"""
        # YOUR CODE STARTS HERE
        if not (self.session and self.event_source and self.session_url):
            raise RuntimeError("Not connected to server")

        req_id = self._id_counter
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_args,
            },
        }
        self._id_counter += 1

        await self.session.post(self.session_url, json=payload)

        async for event in self.event_source:
            raw = event.data.strip()
            # skip any stray session_id messages
            if raw.startswith("/") or raw.startswith("'/"):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # handle only our response
            if msg.get("id") != req_id:
                continue

            result = msg.get("result")
            # server returns a list of strings for tool calls
            if isinstance(result, list) and result:
                return result[0]
            return str(result)
        # YOUR CODE ENDS HERE

    async def cleanup(self):
        """Properly clean up the session and streams"""
        # YOUR CODE STARTS HERE
        if self.event_source:
            try:
                await self.event_source.close()
            except Exception:
                pass
        if self.session:
            await self.session.close()
        # YOUR CODE ENDS HERE



class Defuser(BombClient):
    async def run(self, action: str) -> str:
        """Run a defuser action"""
        # YOUR CODE STARTS HERE
        resp = await self.process_query("game_interaction", {"command": action})

        try:
            data = json.loads(resp)
            if isinstance(data, dict) and 'content' in data:
                resp = data['content'][0]['text']
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(resp)
                if isinstance(data, dict) and 'content' in data:
                    resp = data['content'][0]['text']
            except Exception:
                pass

        if "BOOM!" in resp:
            return "BOOM!"
        if "DISARMED" in resp:
            return "BOMB SUCCESSFULLY DISARMED!"
        #if action.lower() == "state":
            #return resp.splitlines()[0].strip("= ").upper()
        return resp
        # YOUR CODE ENDS HERE


class Expert(BombClient):
    async def run(self) -> str:
        """Run an expert action"""
        # YOUR CODE STARTS HERE
        resp = await self.process_query("get_manual", {})

        try:
            data = json.loads(resp)
            if isinstance(data, dict) and 'content' in data:
                return data['content'][0]['text']
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(resp)
                if isinstance(data, dict) and 'content' in data:
                    return data['content'][0]['text']
            except Exception:
                pass

        if "BOOM!" in resp:
            return "BOOM!"
        if "DISARMED" in resp:
            return "BOMB SUCCESSFULLY DISARMED!"
        return resp.splitlines()[0]



async def main():
    """ Main function to connect to the server and run the clients """
    # YOUR CODE STARTS HERE
    parser = argparse.ArgumentParser(description="Run MCP game client")
    parser.add_argument("--url", required=True, help="Server URL, e.g. http://localhost:8080")
    parser.add_argument("--role", required=True, choices=["Defuser", "Expert"])
    args = parser.parse_args()

    if args.role == "Defuser":
        client = Defuser()
        await client.connect_to_server(args.url)
        try:
            while True:
                action = input("Defuser> ").strip()
                result = await client.run(action)
                print(result)
                if result in ("BOOM!", "BOMB SUCCESSFULLY DISARMED!"):
                    break
        finally:
            await client.cleanup()
    else:
        client = Expert()
        await client.connect_to_server(args.url)
        try:
            while True:
                input("Expert> ")
                result = await client.run()
                print(result)
                if result in ("BOOM!", "BOMB SUCCESSFULLY DISARMED!"):
                    break
        finally:
            await client.cleanup()
    # YOUR CODE ENDS HERE

async def expert_test(expert_client: Expert):
    """Test the Expert class"""
    result = await expert_client.run()
    possible_outputs = ["BOOM!", "BOMB SUCCESSFULLY DISARMED!",
                        "Regular Wires Module", "The Button Module",
                        "Memory Module", "Simon Says Module"]
    assert any(output in result for output in possible_outputs), "Expert test failed"

async def defuser_test(defuser_client: Defuser):
    """Test the Defuser class"""
    result = await defuser_client.run("state")
    assert "BOMB STATE" in result, "Defuser test failed"


if __name__ == "__main__":
    asyncio.run(main())
