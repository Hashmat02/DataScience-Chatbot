####################################################################################################
import os
import regex as re # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np # type: ignore
import openai   # type: ignore
import json
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatAnyscale
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.agents import AgentExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
import gradio as gr

plt.style.use('ggplot')
####################################################################################################


# insert your API key here
os.environ["ANYSCALE_API_KEY"] = "esecret_8btufnh3615vnbpd924s1t3q7p"
os.environ["OPENAI_API_KEY"] = "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"
memory_key = "history"


rag_sys_prompt = """
You are a retriever augmented generation agent. The user gives you a claim and your job is to figure out the \
most relevant hypothesis testing scheme to test this claim. \
The user gives you two things in total:
1. A query.
2. A set of relevant documents.

using the query and relevant documents, simply name the SINGLE BEST HYPOTHESIS Testing scheme for the user claim.

your output must look like this ONLY. NO extra text shall be generated by you!:
name: {testing_scheme}
"""

rag_context = [{'role':'system', 'content':rag_sys_prompt}]

client = openai.OpenAI(
    base_url = "https://api.endpoints.anyscale.com/v1",
    api_key = "esecret_8btufnh3615vnbpd924s1t3q7p"
)

def get_FAISS(doc_path:str = 'Hypothesis_Test_Agent/testing_schemes.pdf'):
    loader = PyPDFLoader(doc_path)
    doc = loader.load()
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(doc)
    db = FAISS.from_documents(texts, embeddings) 

    return db

def merge_docs(docs) :
    output = ""
    
    for doc in docs:
        output += f" {doc.page_content}"
        
    return output

rag_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_test",
            "description": "Returns the most relevant testing scheme.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scheme": {
                        "type": "string",
                        "description": "The name of the most relevant hypothesis testing scheme in the light of the user claim and relevant documents.",
                    }
                },
                "required": ["scheme"],
            },
        },
    }
]

def RAG(user_claim:str =''):
    
    rag_query = f"""
    Given the user claim, which hypothesis test can be used to verify it?
    This is the user query:
    {user_claim}. Verify this claim.
    """
    
    db = get_FAISS()
    retrieved_documents = db.similarity_search(rag_query)
    retrieved_text = merge_docs(retrieved_documents)
    
    user_query = f"User Claim: {user_claim}\nRelevant Documents: {retrieved_text}"
    
    rag_context.append({"role":"user", "content":user_query})
    
    chat_completion = client.chat.completions.create(
        model = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages = rag_context,
        tools=rag_tools,
        tool_choice={"type": "function", "function": {"name": "get_test"}}
    )

    result = chat_completion.model_dump()['choices'][0]['message']['tool_calls'][0]['function']['arguments']
    result = json.loads(result)
    scheme = result['scheme']
    
    return scheme

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="""A Python shell. Shell can dislay charts too. Use this to execute python commands.\
    You have access to all libraries in python including but not limited to sklearn, pandas, numpy,\
    matplotlib.pyplot, seaborn etc. Input should be a valid python command. If the user has not explicitly\
    asked you to plot the results, always print the final output using print(...)\
    Execute all the code.""",
    func=python_repl.run,
)

tools = [repl_tool]


FORMAT_INSTRUCTIONS = """
You must 
1. Claim: The claim made by the user which you need to verify.
2. Null Hypothesis: The null hypothesis from the user claim.
3. Alternate Hypothesis: The alternate hypothesis from the user claim.
4. Test: The hypothesis testing scheme given by the user.
5. Action Input: the input to the action. Use bullet points to format this section.\
6. Observation: Result of the Action executed in Python using the repl_tool.\
7. Thought: I now know the final answer.\
8. Final Answer: Your final accept/reject verdict in light of the results from the Action and Observation. Please include the reason for accept/reject verdict.

** KEEP GENERATING STEPS 1 to 6 UNTIL YOU GENERATE A FINAL ANSWER. DO NOT TERMINATE BEFORE A FINAL ANSWER IS READY.** 
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are a Hypothesis Testing Agent. Your job is to accept/reject user claims about a dataset by using\
            a given hypothesis testing technique. The user gives you the following inputs:
            1. A dataframe.
            2. A Claim about the dataframe (which you need to accept/reject)
            3. A Hypothesis testing scheme which you need for verifying the claim.
            
            Given these inputs, you need to write code to test the claim using the hypothesis testing scheme \
            given by the user.\
            Obey this formatting protocol at all times:
            {FORMAT_INSTRUCTIONS}.\n
            
            WHAT IS THE FINAL ANSWER ?
            """
        ),
        ("user", "Data: {df}\Claim: {input}\nTesting Scheme: {best_test}\ntools:{tools}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def infer_hypothesis(user_input, df, llm):

    agent = (
        {
            "input": lambda x: x["input"],
            "df": lambda x:x['df'],
            "tools": lambda x:tools,
            "best_test": lambda x:best_test,
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )
        }
        | prompt
        | llm
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools = tools, verbose=True)

    generate = True
    best_test = ''
    while generate:
        best_test = RAG(user_input)
        print("Best test:",best_test)
        # fetch the response from the agent`
        result = list(agent_executor.stream({"input": user_input, "df":df, "best_test":best_test},
                                             {"tool": tools}))

        pattern = r"(?i)final\s+answer"

        # Check if the pattern matches the string
        if re.search(pattern, result[-1]['output']):
            generate = False
        else: 
            generate = True
    
    
    return result[-1]['output']
