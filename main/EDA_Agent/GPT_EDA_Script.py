import pandas
import numpy
import os
import sys
import subprocess
import regex as re
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from PIL import Image


def delete_png_files(dir_path):
    for filename in os.listdir(dir_path):
        if filename.endswith(".png"):
            os.remove(os.path.join(dir_path, filename))


def run_code(code):

    with open(f'{os.getcwd()}/EDA_Agent/code/code.py', 'w') as file:
        file.write(code)

    try:
        print("Running code ...\n")
        result = subprocess.run([sys.executable, 'EDA_Agent/code/code.py'],
                                capture_output=True, text=True, check=True, timeout=20)
        return result.stdout, False

    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr, True

    except subprocess.TimeoutExpired:
        return "Execution timed out.", True


# Always uses GPT 3.5 Turbo
def inferEDAGPT(user_query: str):
    tools = [PythonREPLTool()]
    llm = ChatOpenAI(temperature=0)  # Defaults to GPT 3.5 Turbo
    # Bind the tools to the llm.
    # llm_with_tools = llm.bind_tools(tools)
    # Construct the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are Data Analysis assistant. Your job is to use your tools to answer a user query in the best\
                manner possible. Your job is to respond to user queries by generating code in a .py file and using it to answer the question.\
                In case you make plots, make sure to label the axes and add a good title too. DO NOT SHOW THE PLOTS. JUST SAVE THEM! \
                You must save any plots in the 'EDA_Agent/graphs' folder as png only.\
                Provide no explanation for your code.\
                Read the data from 'df.csv'.\
                """,
            ),
            ("user", "Query: {input}\n Error: {error}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # Clean up images from previous runs.
    image_path = f'{os.getcwd()}/EDA_Agent/graphs'
    delete_png_files(image_path)
    # Create the agent.
    agent = (
        {
            "input": lambda x: x["input"],
            "error": lambda x: x["error"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # Running the code
    error = True
    prev_output = None
    while error:
        result = agent_executor.invoke(
            {"input": user_query, "error": prev_output})
        pattern = r"```python\n(.*?)\n```"
        code = re.findall(pattern, result["output"], re.DOTALL)
        code = "\n".join(code)
        prev_output, error = run_code(code)
        print("Code outputted: ", prev_output)
    # fetch image
    image = None
    for file in os.listdir(image_path):
        if file.endswith(".png"):
            image = numpy.array(Image.open(f'{image_path}/{file}'))
            break
    return result["output"] + "\n" + prev_output, image
