## FINETUNING Runthrough

For our super inference agent, we decided to finetune an already existing model on our own data. The entire process in handled through the [axolotl wrapper](https://github.com/OpenAccess-AI-Collective/axolotl).

The model we chose for finetuning is [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), there are several reasons for this choice:

<ol>
    <li>It is an open-source model, that is readily accessible through huggingface.</li>
    <li>Computational requirements were within limits of our available resources.</li>
    <!-- <li></li> -->
    <li>It is an instruct fine-tuned model already.</li>
    <li>Larger context window.</li>
</ol>

---

### The dataset

To finetune according to our super inference agent requirement, we required very specific data. Due to this, we assigned three members of our team to solely focus on curating this dataset.

We wanted our super inference agent to be able to handle multiple types of architectures and for this reason we made sure to have some diversity in our dataset. Our dataset contains the following architectures:

<ul>
    <li>Transformers</li>
    <li>LSTMS</li>
    <li>RNNs</li>
    <li>FeedForwrad NNs</li>
    <li>CNNs</li>
</ul>

Our dataset follows alpaca format as this was the most common approach we found for finetuning instruct models. Each data sample contains an instruction, input, and output:

- The instruction contains the system prompt (defined in sys_prompt.txt) as well as the user prompt which defines the task to perform, the type and dimension of each layer, activation functions to use and finally the path to the data.
- The input is left empty as it is not a requiremnt in the alpaca format.
- The output is the entire code corresponding to the architecture details defined in the input.

In total, our dataset contains 36 samples. The dataset in it's entirety is present in the **data.jsonl** file.

---

### QLoRA

Due to our limited computational resources, we decided to use the QLoRA apporach for effecient finetuning.

### The process

As menitoned earlier the enitre finetuning process was run through the axolotl wrapper. The wrapper provides a cli through which we can train the model. All parameters for the finetuning are defined in the **qlora.yml** file.

We ran the model for **18 epochs** with a **mini batch size of 2** in order to cover our data set entirely once.

The entire code for the finetuning process is present in the **finetuning.ipynb** notebook.

---

### The results

After the training was complete the adpater was saved in a directory defined in the **qlora.yml** configuration file, and also pushed it to huggingface.

The results of the training process are shown below:

| Training Loss | Epoch | Step | Validation Loss |
| :-----------: | :---: | :--: | :-------------: |
|    0.2077     |  1.0  |  1   |     0.3832      |
|    0.1985     |  2.0  |  2   |     0.3782      |
|    0.1979     |  2.0  |  3   |     0.3553      |
|    0.1901     |  3.0  |  4   |     0.3147      |
|    0.1624     |  3.0  |  5   |     0.2756      |
|    0.1489     |  4.0  |  6   |     0.2525      |
|    0.1342     |  4.0  |  7   |     0.2383      |
|    0.1137     |  5.0  |  8   |     0.2228      |
|    0.1026     |  5.0  |  9   |     0.2040      |
|    0.1001     |  6.0  |  10  |     0.1905      |
|    0.0828     |  6.0  |  11  |     0.1816      |
|    0.0746     |  7.0  |  12  |     0.1751      |
|    0.0687     |  7.0  |  13  |     0.1707      |
|    0.0544     |  8.0  |  14  |     0.1654      |
|    0.0526     |  8.0  |  15  |     0.1620      |
|    0.0469     |  9.0  |  16  |     0.1591      |
|     0.048     |  9.0  |  17  |     0.1575      |
|    0.0392     | 10.0  |  18  |     0.1573      |

---

### Inference

The adapter for our model is available at [Burhan02/Mistral-7B-Instruct-v0.2-ft](https://huggingface.co/Burhan02/Mistral-7B-Instruct-v0.2-ft). To run inference on our adapter, it first needs to be merged with the base model. The entire process has been clearly defined in the **inference.ipynb** notebook and **inference.py** file. The results of inference on three seperate inputs is stored in **output_1.py, output_2.py, output_3.py**
