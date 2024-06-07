import os
import json
from huggingface_hub import login
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MistralForCausalLM,
    LlamaTokenizer,
)
from peft import PeftModel


def main():

    auth_token = "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"
    login(token=auth_token)

    os.environ["HF_TOKEN"] = auth_token

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    peft_model = "Burhan02/Mistral-7B-Instruct-v0.2-ft"

    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, peft_model)

    model.merge_and_unload()
    model.to("cuda")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    def infer(i, user_prompt):
        sys_prompt = """
        You are an agent responsible for building and runnning nerual networks solely in the PyTorch library.
        The user gives you 3 inputs:
        1. Layers: this is the layer by layer breakdown of the neural network. This includes things like the dimensionality 
        of the (input size, output size).
        2. Task: This is either regression, binary classification or multiclass-classification 
        (with the number of classes specified by the user).
        3. Data Path: local path to the data.csv file (always a csv).

        Given these inputs, your job is to generate a .py file that contains the following:
        1. A neural_net class in pytorch which includes these methods:
            a. init
            b. forward 
            c. train
            d. predict
        2. Code to read in the dataframe from data_path given by the user.
        3. Code to split the dataframe into train and test set according to the user task.
        4. Code to train the the neural net you created to train on train set and predict on test set.

        Make sure to enclose your output in ''' at the beginning and end. ONLY GENERATE THE .py CODE FILE CONTAINING THE PYTORCH CODE.
        DO NOT EXPLAIN YOURSELF.
        """

        updated_prompt = sys_prompt + user_prompt
        inputs = tokenizer(updated_prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_length=6000)

        outputs = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        with open(f"output_{i}.py", "w") as file:
            code = outputs.split("'''")[2]
            file.write(code)
            file.close()

    user_prompt = """
    Layers:
    - Fully connected layer 1:
    - Input size: 10
    - Output size: 128
    - Fully connected layer 2:
    - Input size: 128
    - Output size: 256
    - Fully connected layer 3:
    - Input size: 256
    - Output size: 128
    - Fully connected layer 4 (readout layer):
    - Input size: 128
    - Output size: 4

    Activation function between hidden layers except last:
        - ReLU

    Apply softmax to of size 4 after output layer
        
    Task:
        - Multiclass Classification with 4 classes.

    Data Path: "./root/data.csv"
    """

    infer(1, user_prompt)


if __name__ == "__main__":
    main()
