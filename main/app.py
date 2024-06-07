import gradio as gr
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from langchain_community.chat_models import ChatAnyscale
from Hypothesis_Test_Agent.Hypothesis_Agent import infer_hypothesis
from EDA_Agent.EDA import infer_EDA
from Basic_Inf_Agent.Basic_Inference_Agent import infer_basic
from EDA_Agent.GPT_EDA_Script import inferEDAGPT
from Vanilla_Super_Inf_Agent.GPT_SuperInf import inferGPTVanillaSuperInf

print(os.getcwd())


def run_agent(data_path: str = '', provider: str = 'Mistral', agent_type: str = 'EDA', query: str = '', layers: str = '', temp: float = 0.1):

    print(data_path)
    df = pd.read_csv(data_path)
    df.to_csv('./df.csv', index=False)

    if provider.lower() == 'mistral':

        llm = ChatAnyscale(
            model_name='mistralai/Mixtral-8x7B-Instruct-v0.1', temperature=temp)

        if agent_type == 'Data Explorer':
            EDA_response, EDA_image = infer_EDA(
                user_input=query, llm=llm, df=df)
            return EDA_response, EDA_image

        if agent_type == 'Hypothesis Tester':
            hypothesis_response = infer_hypothesis(
                user_input=query, llm=llm, df=df)
            return hypothesis_response, None

        if agent_type == 'Basic Inference':
            basic_response = infer_basic(user_input=query, df=df, llm=llm)
            return basic_response, None

    if provider.lower() == "gpt":
        if agent_type == "Super Inference":
            super_inf_response = inferGPTVanillaSuperInf(query + "\n" + layers)
            return super_inf_response, None
        if agent_type == "Hypothesis Tester":
            return "Hypothesis Tester agent is only available via the Mistral LLM Provider", None
        else:
            eda_basic_response, image = inferEDAGPT(query)
            return eda_basic_response, image

    return None


demo = gr.Interface(
    run_agent,
    [
        gr.UploadButton(label="Upload your CSV!"),
        gr.Radio(["Mistral", "GPT"], label="Select Your LLM"),
        gr.Radio(["Data Explorer", "Hypothesis Tester", "Basic Inference",
                 "Super Inference"], label="Choose Your Agent"),
        gr.Text(label="Query", info="Your input to the Agent. Be descriptive!"),
        gr.Text(label="Architecture",
                info="Specify the layer by layer architecture only for Super Inference Agent"),
        gr.Slider(label="Model Temperature",
                  info="Slide right to make your model more creative!")
    ],
    [
        gr.Text(label="Agent Output",
                info="This might take a while to generate since agent debugs its errors too ..."),
        gr.Image(label="Graph")
    ]
)
demo.launch(share=True)
