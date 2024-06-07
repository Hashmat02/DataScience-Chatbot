import os
import re
import subprocess
import sys
from PIL import Image
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatAnyscale
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
plt.style.use('ggplot')
####################################################################################################

os.environ["ANYSCALE_API_KEY"] = "esecret_8btufnh3615vnbpd924s1t3q7p"
memory_key = "history"

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""You are Data Analysis assistant. Your job is to use your tools to answer a user query in the best\
            manner possible. Your job is to respond to user queries by generating a python code file.\
            Make sure to include all necessary imports.\
            In case you make plots, make sure to label the axes and add a good title too. \
            You must save any plots in the 'EDA_Agent/graphs' folder as png only.\
            Provide no explanation for your code.\
            Read the data from './df.csv'.\
            ** Enclose all your code between triple backticks ``` **\
            RECTIFY ANY ERRORS FROM THE PREVIOUS RUNS.
            """,
        ),
        ("user", "Dataframe named df: {df}\nQuery: {input}\nTools:{tools}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="""A Python shell. Shell can dislay charts too. Use this to execute python commands.\
    You have access to all libraries in python including but not limited to sklearn, pandas, numpy,\
    matplotlib.pyplot, seaborn etc. Input should be a valid python command. If the user has not explicitly\
    asked you to plot the results, always print the final output using print(...)""",
    func=python_repl.run,
)

if 'code' not in os.listdir():
    os.mkdir('code')
    
tools = [repl_tool]


def delete_png_files(dir_path):
    for filename in os.listdir(dir_path):
        if filename.endswith(".png"):
            os.remove(os.path.join(dir_path, filename))

def run_code(code):
    
    with open(f'{os.getcwd()}/EDA_Agent/code/code.py', 'w') as file:
        file.write(code)
        
    try:
        print("Running code ...\n")
        result = subprocess.run([sys.executable, 'EDA_Agent/code/code.py'], capture_output=True, text=True, check=True, timeout=20)
        return result.stdout, False
    
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr, True
    
    except subprocess.TimeoutExpired:
        return "Execution timed out.", True

def infer_EDA(llm=None, user_input:str = '', df=''):    
    agent = (
            {
                "input": lambda x: x["input"],
                "tools": lambda x:x['tools'],
                "df": lambda x:x['df'],
                "prev_error":lambda x:x['prev_error'],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                )
            }
            | prompt
            | llm
            | OpenAIToolsAgentOutputParser()
        )

    EDA_executor = AgentExecutor(agent=agent, tools=tools, df = df, prev_error='', verbose=True)
    
    # Running Inference
    error_flag = True
    image_path = f'{os.getcwd()}/EDA_Agent/graphs'
    
    delete_png_files(image_path)
    res = None
    
    while error_flag:
        result = list(EDA_executor.stream({"input": user_input, 
                                     "df":df,
                                     "prev_error":res,
                                     "tools":tools}))

        # need to extract the code
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, result[0]['output'], re.DOTALL)
        code = "import matplotlib\nmatplotlib.use('Agg')\n"
        code += "\n".join(matches)

        # execute the code
        res, error_flag = run_code(code)
        print(res)
    
    image = None
    
    for file in os.listdir(image_path):
        if file.endswith(".png"):
            image = np.array(Image.open(f'{image_path}/{file}'))
            break
    
    final_val = result[-1]['output'] + f'\n{res}'
    
    return final_val , image

