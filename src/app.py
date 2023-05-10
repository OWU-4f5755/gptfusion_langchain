from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI

import gradio as gr
import sys
import os
import json
import argparse

def construct_index(directory_path, api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    max_input_size = 8192  # 8192 max tokens in PROMPT with gpt-4; 4096 input tokens originally with gpt-3.5-turbo
    num_outputs = 1024  # 1024 = 1024/8 max tokens in ANSWER; originally 512 output tokens with gpt-3.5-turbo (but maintained 8:1 ratio)
    max_chunk_overlap = 20  # 20 = max tokens to overlap between chunk 1 and chunk 2
    chunk_size_limit = 600  # 600 = max tokens to put in a single chunk

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.4, model_name="gpt-4", max_tokens=num_outputs))  # delta t=0.4 <- t=0.7 (generally 0.2-0.5 for medicine) && delta model_name="gpt-4-32k" <- model_name="gpt-3.5-turbo"

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

conversation_history = {
    '1': [],
    '2': []
}

def chatbot(input_text1, input_text2, access_history=False):
    global conversation_history
    if access_history:
        # Load conversation history from disk
        with open('conversation_history.json', 'r') as f:
            conversation_history = json.load(f)

    response1 = None
    response2 = None

    if input_text1.strip():  # check if input_text1 is not empty
        conversation_history['1'].append(f"User1: {input_text1}\n")
        input_with_history1 = "".join(conversation_history['1'])
        response1 = index.query(input_with_history1, response_mode="compact")
        conversation_history['1'].append(f"AI: {response1.response}\n")

    if input_text2.strip():  # check if input_text2 is not empty
        conversation_history['2'].append(f"User2: {input_text2}\n")
        input_with_history2 = "".join(conversation_history['2'])
        response2 = index.query(input_with_history2, response_mode="compact")
        conversation_history['2'].append(f"AI: {response2.response}\n")

    # save conversation hx to disk
    with open('conversation_history.json', 'w') as f:
        json.dump(conversation_history, f)

    # return responses, uses empty string if the response is None
    return (response1.response if response1 else "", response2.response if response2 else "")


iface = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.components.Textbox(lines=15, label="Enter your text (1)", placeholder="Type your message here..."),
        gr.components.Textbox(lines=15, label="Enter your text (2)", placeholder="Type your message here..."),
        gr.components.Checkbox(label="Access other conversation history")
    ],
    outputs=[
        gr.components.Text(label="text (1)"),
        gr.components.Text(label="text (2)")
    ],
    title="A GPT-4 AI on Your Custom Files"
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='A GPT-4 AI on Your Custom Files')
    parser.add_argument('--api-key', required=True, help='The OpenAI API key')
    args = parser.parse_args()

    index = construct_index("docs", args.api_key)
    iface.launch(share=True)

# from llama_index import SimpleDirectoryReader, ServiceContext, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
# from langchain.chat_models import ChatOpenAI
# from langchain import OpenAI
# import gradio as gr
# import sys
# import os

# os.environ["OPENAI_API_KEY"] = ''

# def construct_index(directory_path):
#     max_input_size = 12228  # 12228 = 4096*3 with gpt-4-32k, but can technically be 32768; 4096 input tokens originally with gpt-3.5-turbo
#     num_outputs = 10240  # 10240 = 512*20; originally 512 output tokens with gpt-3.5-turbo code
#     max_chunk_overlap = 20
#     chunk_size_limit = 600

#     # prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

#     # llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.4, model_name="gpt-4-32k", max_tokens=num_outputs))  # delta t=0.4 <- t=0.7 (generally 0.2-0.5 for medicine) && delta model_name="gpt-4-32k" <- model_name="gpt-3.5-turbo"

#     # documents = SimpleDirectoryReader(directory_path).load_data()

#     # index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
#     prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
#     llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.4, model_name="gpt-4-32k", max_tokens=num_outputs))  # delta t=0.4 <- t=0.7 (generally 0.2-0.5 for medicine) && delta model_name="gpt-4-32k" <- model_name="gpt-3.5-turbo"
#     service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
#     index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
#     index.save_to_disk('index.json')

#     return index

# conversation_history = []

# def chatbot(input_text):
#     global conversation_history
#     conversation_history.append(f"User: {input_text}\n")

#     input_with_history = "".join(conversation_history)
#     # index = GPTSimpleVectorIndex.load_from_disk('index.json')  # original chatbot() code
#     index = load_index_from_storage(service_context=service_context)
#     response = index.query(input_with_history, response_mode="compact")  # original chatbot() code

#     conversation_history.append(f"AI: {response.response}\n")

#     return response.response  # original chatbot() code

# iface = gr.Interface(
#     fn=chatbot,
#     inputs=gr.components.Textbox(lines=15, label="Enter your text", placeholder="Type your message here..."),
#     outputs=gr.components.Text(label="text"),
#     title="GPT-4 AI Chatbot on Custom Files"
# )

# index = construct_index("docs")
# iface.launch(share=True)
