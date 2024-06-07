import pandas
import numpy
import subprocess
import os
import sys
import regex as re
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)


def run_code(code):

    with open(f'{os.getcwd()}/Vanilla_Super_Inf_Agent/code/code.py', 'w') as file:
        file.write(code)

    try:
        print("Running code ...\n")
        result = subprocess.run([sys.executable, 'Vanilla_Super_Inf_Agent/code/code.py'],
                                capture_output=True, text=True, check=True, timeout=20)
        return result.stdout, False

    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr, True

    except subprocess.TimeoutExpired:
        return "Execution timed out.", True


# Not using pre-built tools, I'll execute the code myself.
def inferGPTVanillaSuperInf(user_query: str):
    tools = [PythonREPLTool()]
    llm = ChatOpenAI(temperature=0)  # Defaults to GPT 3.5 Turbo
    # Bind the tools to the llm.
    # llm_with_tools = llm.bind_tools(tools)
    # Construct the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an agent responsible for building and runnning nerual networks solely in the 
                PyTorch library.
                The user gives you 3 inputs:
                1. Layers: this is the layer by layer breakdown of the neural network. This includes things like the dimensionality 
                of the (input size, output size).
                2. Task: This is either regression, binary classification or multiclass-classification 
                (with the number of classes specified by the user).

                Always read data from df.csv. Given these inputs, your job is to generate a .py file that contains the following:
                1. A neural_net class in pytorch which includes these methods:
                    a. init
                    b. forward 
                    c. train
                    d. predict
                2. Code to read in the dataframe from data_path given by the user.
                3. Code to split the dataframe into train and test set according to the user task.
                4. Code to train the the neural net you created to train on train set and predict on test set.
                5. Code to report the training accuracy.
                ONLY OUTPUT CODE!
                RECTIFY ANY PREVIOUS ERRORS.
                """,
            ),
            ("user", "Query: {input} Error: {error}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # prompt = hub.pull("hwchase17/openai-tools-agent")
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

    return result["output"] + "\n" + prev_output


# if __name__ == "__main__":
#     stop = False
#     while not stop:
#         query = input("Ask a question regarding the dataframe: ")
#         if query == "Q" or query == "q":
#             stop = True
#             continue
#         print(inferGPTVanillaSuperInf(query))

# '''
# Layers:
# layer1: (10, 100)
# layer2: (100, 200)
# layer3: (200, 100)
# layer4: (100, 1)
# Task:
# Binary classification to predict the probability of survival. You can drop the PassengerID and Name features.
# Data Path:
# df.csv
# '''
