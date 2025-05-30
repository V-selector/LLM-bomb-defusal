from ml_collections import ConfigDict
from crewai import Agent, Crew, Task, LLM, Process
from crewai_bomb.tools import DefuserTool, ExpertTool
import warnings

warnings.filterwarnings("ignore", message="Unclosed client session")

model_config = ConfigDict()
model_config.extra = 'allow'

llm = LLM(
    model='ollama/qwen2.5', 
    model_config=model_config
)

defuser_tool = DefuserTool()
expert_tool = ExpertTool()

defuser_agent = Agent(
    name='DefuserAgent',
    role='Defuser',
    goal='Use only bomb commands like: state, cut wire X, press, hold, release on X',
    backstory='You see the bomb but not the manual.',
    llm=llm,
    tools=[defuser_tool]
)

expert_agent = Agent(
    name='ExpertAgent',
    role='Expert',
    goal='Given module state, return one valid bomb command only. No explanation.',
    backstory='You have the manual but cannot see the bomb.',
    llm=llm,
    tools=[expert_tool]
)

if __name__ == '__main__':
    while True:
        print(f"\nStep: Defuser reads module state")
        task_state = Task(
            agent=defuser_agent,
            tool=defuser_tool,
            input={"command": "state"},
            description="Get bomb status using the 'state' command.",
            expected_output="Description of bomb state"
        )
        crew_state = Crew(
            agents=[defuser_agent, expert_agent],
            tasks=[task_state],
            process=Process.sequential,
            verbose=True
        )
        state = crew_state.kickoff().raw.strip()
        print(f"\n[Defuser sees state]:\n{state}\n")

        if "BOMB EXPLODED" in state.upper() or "BOMB SUCCESSFULLY DISARMED" in state.upper():
            print("Game has ended. Restart server if needed.")
            break

        print(f"Step: Expert reads state and suggests action")
        task_manual = Task(
            agent=expert_agent,
            tool=expert_tool,
            input={"state": state},
            description="Based on the state, return one valid disarm command only.",
            expected_output="cut wire 2, hold, press, etc."
        )
        crew_manual = Crew(
            agents=[defuser_agent, expert_agent],
            tasks=[task_manual],
            process=Process.sequential,
            verbose=True
        )
        instruction = crew_manual.kickoff().raw.strip()
        print(f"\n[Expert suggests]: {instruction}\n")

        print(f"Step: Defuser executes the suggestion")
        task_action = Task(
            agent=defuser_agent,
            tool=defuser_tool,
            input={"command": instruction},
            description="Run the Expert's instruction.",
            expected_output="Result of the bomb action"
        )
        crew_action = Crew(
            agents=[defuser_agent, expert_agent],
            tasks=[task_action],
            process=Process.sequential,
            verbose=True
        )
        result = crew_action.kickoff().raw.strip()
        print(f"\n[Defuser executes]: {result}\n")

        if "BOMB EXPLODED" in result.upper() or "BOMB SUCCESSFULLY DISARMED" in result.upper():
            print("Final outcome received. Exiting.")
            break
