from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")


tasks = []


@tool
def add_task(task_name: str) -> str:
    """
    Add a new task to the task list.
    """
    tasks.append(task_name)
    return f"Task '{task_name}' added successfully."


@tool
def view_tasks(dummy: str = "") -> str:
    """
    Display all available tasks.
    """

    if not tasks:
        return "No tasks available."

    return "\n".join(
        [f"{i+1}. {task}" for i, task in enumerate(tasks)]
    )


@tool
def search_task(keyword: str) -> str:
    """
    Search tasks using a keyword.
    """

    results = [
        task
        for task in tasks
        if keyword.lower() in task.lower()
    ]

    if not results:
        return "No matching tasks found."

    return "\n".join(
        [f"{i+1}. {task}" for i, task in enumerate(results)]
    )

 

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)


tools = [
    add_task,
    view_tasks,
    search_task
]



prompt = hub.pull("hwchase17/react")



agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)



agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=1,
    early_stopping_method="force"
)



print("===================================")
print("      TASK MANAGER AGENT")
print("===================================")
print("Commands Examples:")
print("- Add task Prepare AI Presentation")
print("- View tasks")
print("- Search task AI")
print("- Exit")
print("===================================\n")

while True:

    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:

        response = agent_executor.invoke(
            {"input": user_input}
        )

        print("\nAgent:")
        print(response["output"])
        print()

    except Exception as e:
        print("\nError:")
        print(e)
        print()