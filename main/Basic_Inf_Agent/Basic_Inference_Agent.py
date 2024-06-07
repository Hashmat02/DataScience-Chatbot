####################################################################################################
import os
import re
import subprocess
import sys
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

plt.style.use('ggplot')
####################################################################################################

os.environ["ANYSCALE_API_KEY"] = "esecret_8btufnh3615vnbpd924s1t3q7p"
memory_key = "history"


python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="""A Python shell. Shell can dislay charts too. Use this to execute python commands.\
    You have access to all libraries in python including but not limited to sklearn, pandas, numpy,\
    matplotlib.pyplot, seaborn etc. Input should be a valid python command. If the user has not explicitly\
    asked you to plot the results, always print the final output using print(...)""",
    func=python_repl.run,
)

tools = [repl_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Machine Learning Inference agent. Your job is to use your tools to answer a user query\
            in the best manner possible.\
            The user gives you two things:
            1. A query, and a target task.
            Your job is to extract what you need to do from this query and then accomplish that task.\
            In case the user asks you to do things like regression/classification, you will need to follow this:\
            IMPORT ALL NECESSARY LIBRARIES\
            READ IN YOUR DATAFRAME FROM: './df.csv'
            1. Split the dataframe into X and y (y= target, X=features)
            2. Perform some preprocessing on X as per user instructions. 
            3. Split preprocessed X and y into train and test sets.
            4. If a model is specified, use that model or else pick your model of choice for the given task.
            5. Run this model on train set. 
            6. Predict on Test Set.
            5. Print accuracy score by predicting on the test.
            
            Provide no explanation for your code. Enclose all your code between triple backticks ``` """,
        ),
        ("user", "Dataframe named df:\n{df}\nQuery: {input}\nList of Tools: {tools}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

if 'code' not in os.listdir():
    os.mkdir('code')


def run_code(code):
    
    with open(f'{os.getcwd()}/Basic_Inf_Agent/code/code.py', 'w') as file:
        file.write("import os\nprint(f'{os.getcwd()}')\n")
        file.write(code)
        
    try:
        print("Running code ...\n")
        result = subprocess.run([sys.executable, f'{os.getcwd()}/Basic_Inf_Agent/code/code.py'], capture_output=True, text=True, check=True, timeout=20)
        return result.stdout, False
    
    except subprocess.CalledProcessError as e:
        print("Error in code!\n")
        return e.stdout + e.stderr, True
    
    except subprocess.TimeoutExpired:
        return "Execution timed out.", True


def infer_basic(user_input, df, llm):

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
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, df = df, prev_error='', verbose=True)
    
    error_flag = True    
    res = None

    while error_flag:
        result = list(agent_executor.stream({"input": user_input, 
                                         "df":pd.read_csv('df.csv'), 
                                         "prev_error": res,
                                         "tools":tools}))
        
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, result[0]['output'], re.DOTALL)
        # code = "import os\nos.chdir({os.getcwd()}/Basic_Inf_Agent)\n"
        code = "\n".join(matches)
        code += "\ndf.to_csv('./df.csv', index=False)"
        
        res, error_flag = run_code(code)
        
        print(res)
            
    # execute the code
    return res

