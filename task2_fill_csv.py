import csv
import asyncio
from ml_collections import ConfigDict
from game_mcp.game_client import Defuser
from crewai import LLM

PROMPTS = {
    "text":     "You are the Expert. The defuser reports: {state}. What is the next action?",
    "markdown": "### Observation\n{state}\n\n### Instructions\nWhat should the defuser do?",
    "json":     '{{"role":"Expert","input":{{"state":"{state}"}}}}'
}

TEMPS = [0.1, 0.9]
TOP_PS = [0.8]
TOP_KS = [20, 40, 60]

VALID_COMMANDS = ["cut", "press", "hold"]

def get_llm(temp, top_p, top_k):
    cfg = ConfigDict()
    cfg.extra = "allow"
    args = {
        "provider": "custom",
        "custom_llm_provider": "ollama",
        "model": "qwen2.5:1.5b",
        "base_url": "http://localhost:11434",
        "temperature": temp,
        "top_p": top_p,
        "model_config": cfg
    }
    if top_k is not None:
        args["top_k"] = top_k
    return LLM(**args)


async def test_config(prompt_style, temp, top_p, top_k):
    llm = get_llm(temp, top_p, top_k)
    defuser = Defuser()
    await defuser.connect_to_server("http://localhost:8080")

    steps = 0
    errors = 0
    success = False

    try:
        while True:
            state = await defuser.run("state")
            if "BOOM!" in state or "DISARMED" in state:
                success = ("DISARMED" in state)
                break

            prompt = PROMPTS[prompt_style].format(state=state)
            instr = llm.call(prompt).strip()

            if not instr or not any(cmd in instr.lower() for cmd in VALID_COMMANDS):
                errors += 1
                continue

            try:
                result = await defuser.run(instr)
                steps += 1
                if "BOOM!" in result or "DISARMED" in result:
                    success = ("DISARMED" in result)
                    break
            except Exception:
                errors += 1
                break

        return success, steps, errors

    finally:
        await defuser.cleanup()


async def main():
    with open("/Users/sandra/desktop/xyy.lee15/llm-bomb-defusal/task2_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_style", "temperature", "top_p", "top_k", "success", "steps", "errors"])
        for style in PROMPTS:
            for temp in TEMPS:
                for p in TOP_PS:
                    for k in TOP_KS:
                        print(f"\n=== Testing config: {style} | temp={temp} | top_p={p} | top_k={k} ===")
                        try:
                            s, st, e = await test_config(style, temp, p, k)
                            writer.writerow([style, temp, p, k, int(s), st, e])
                            print(f"Result: success={s}, steps={st}, errors={e}")
                        except Exception as ex:
                            writer.writerow([style, temp, p, k, 0, 0, 1])
                            print(f"Failed config {style}-{temp}-{p}-{k}: {ex}")

if __name__ == "__main__":
    asyncio.run(main())
