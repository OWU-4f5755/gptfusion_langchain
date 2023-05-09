from llama_index import SimpleDirectoryReader, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import gradio as gr
import sys
import os
import json

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 32768
    num_outputs = 10240
    max_chunk_overlap = 20
    chunk_size_limit = 400

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.4, model_name="gpt-4-32k", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index_creator = VectorStoreIndexCreator(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    # Note: Adjust the index_type and index_params as needed for your specific use case.
    index = index_creator.create_index(index_type='hnsw', index_params={'M': 64, 'ef_construction': 200, 'post': 2})

    index.save_to_disk('index.json')

conversation_history1 = []
conversation_history2 = []

# def chatbot(input_text, access_other_history, chatbot_number):
#     global conversation_history1
#     global conversation_history2

#     if chatbot_number == 1:
#         if access_other_history:
#             conversation_history1.extend(conversation_history2)
#             conversation_history2.clear()
#         return process_chatbot(input_text, conversation_history1)
#     elif chatbot_number == 2:
#         if access_other_history:
#             conversation_history2.extend(conversation_history1)
#             conversation_history1.clear()
#         return process_chatbot(input_text, conversation_history2)

# def process_chatbot(input_text, conversation_history):
#     conversation_history.append(f"User: {input_text}\n")

#     input_with_history = "".join(conversation_history)

#     if len(input_with_history) > max_input_size:
#         response_text = "The input size, including conversation history, exceeds the maximum allowed input size. Please reduce the input or conversation history."
#         conversation_history.append(f"AI: {response_text}\n")
#         return response_text

#     index = VectorStoreIndexWrapper.load_from_disk('index.json')
#     response = index.query(input_with_history, response_mode="compact")

#     conversation_history.append(f"AI: {response.response}\n")

#     return response.response

def chatbot(input_text, access_other_history, chatbot_number, show_sources):
    if chatbot_number == 1:
        if access_other_history:
            conversation_history1.extend(conversation_history2)
            conversation_history2.clear()
        return process_chatbot(input_text, conversation_history1, show_sources)
    elif chatbot_number == 2:
        if access_other_history:
            conversation_history2.extend(conversation_history1)
            conversation_history1.clear()
        return process_chatbot(input_text, conversation_history2, show_sources)

def process_chatbot(input_text, conversation_history, show_sources):
    conversation_history.append(f"User: {input_text}\n")

    input_with_history = "".join(conversation_history)

    if len(input_with_history) > max_input_size:
        response_text = "The input size, including conversation history, exceeds the maximum allowed input size. Please reduce the input or conversation history."
        conversation_history.append(f"AI: {response_text}\n")
        return response_text

    index = VectorStoreIndexWrapper.load_from_disk('index.json')

    if show_sources:
        response = index.query_with_sources(input_with_history)

        formatted_response = f"{response.response}\n\nSources:\n"
        for source in response.sources:
            formatted_response += f"{source}\n"
    else:
        response = index.query(input_with_history, response_mode="compact")
        formatted_response = response.response

    conversation_history.append(f"AI: {formatted_response}\n")

    return formatted_response

iface1 = gr.Interface(
    fn=lambda input_text, access_other_history, show_sources: chatbot(input_text, access_other_history, 1, show_sources),
    inputs=[
        gr.Textbox(lines=15, label="Enter your text (1)", placeholder="Type your message here..."),
        gr.Checkbox(label="Access other conversation history"),
        gr.Checkbox(label="Show sources")  # checkbox component to enable chatbot1_with_sources
    ],
    outputs=gr.Text(label="text (1)"),
    title="A GPT-4 AI on Your Custom Files"
)

iface2 = gr.Interface(
    fn=lambda input_text, access_other_history, show_sources: chatbot(input_text, access_other_history, 2, show_sources),
    inputs=[
        gr.Textbox(lines=15, label="Enter your text (2)", placeholder="Type your message here..."),
        gr.Checkbox(label="Access other conversation history"),
        gr.Checkbox(label="Show sources")  # checkbox component to enable chatbot2_with_sources
    ],
    outputs=gr.Text(label="text (2)"),
    title="A GPT-4 AI on Your Custom Files"
)

parallel_iface = gr.Parallel(iface1, iface2)

index = construct_index("docs")
parallel_iface.launch(share=True)


# from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, PromptHelper  # removed GPTSimpleVectorIndex
# from langchain.chat_models import ChatOpenAI  # old
# from langchain.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.document_loaders import UnstructuredPDFLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import Chroma

# import gradio as gr
# #import gradio.mix  # allow for parallelization in Gradio interface
# import sys
# import os
# import glob  # may not need this
# import json

# os.environ["OPENAI_API_KEY"] = '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'

# def construct_index(directory_path):
#     max_input_size = 32768
#     num_outputs = 10240
#     max_chunk_overlap = 20
#     chunk_size_limit = 400

#     prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

#     llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.4, model_name="gpt-4-32k", max_tokens=num_outputs))

#     documents = SimpleDirectoryReader(directory_path).load_data()

#     index_creator = VectorStoreIndexCreator(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

#     # Note: Adjust the index_type and index_params as needed for your specific use case.
#     index = index_creator.create_index(index_type='hnsw', index_params={'M': 64, 'ef_construction': 200, 'post': 2})

#     index.save_to_disk('index.json')

#     return index

# conversation_history1 = []
# conversation_history2 = []

# def chatbot1(input_text, access_other_history):
#     global conversation_history1
#     global conversation_history2

#     if access_other_history:
#         conversation_history1.extend(conversation_history2)
#         conversation_history2.clear()

#     return process_chatbot(input_text, conversation_history1)

# def chatbot2(input_text, access_other_history):
#     global conversation_history1
#     global conversation_history2

#     if access_other_history:
#         conversation_history2.extend(conversation_history1)
#         conversation_history1.clear()

#     return process_chatbot(input_text, conversation_history2)

# def process_chatbot(input_text, conversation_history):
#     conversation_history.append(f"User: {input_text}\n")

#     input_with_history = "".join(conversation_history)

#     if len(input_with_history) > max_input_size:
#         response_text = "The input size, including conversation history, exceeds the maximum allowed input size. Please reduce the input or conversation history."
#         conversation_history.append(f"AI: {response_text}\n")
#         return response_text

#     index = VectorStoreIndexWrapper.load_from_disk('index.json')
#     response = index.query(input_with_history, response_mode="compact")

#     conversation_history.append(f"AI: {response.response}\n")

#     return response.response

# def chatbot1_with_sources(input_text, access_other_history):
#     global conversation_history1
#     global conversation_history2

#     if access_other_history:
#         conversation_history1.extend(conversation_history2)
#         conversation_history2.clear()

#     return process_chatbot_with_sources(input_text, conversation_history1)

# def chatbot2_with_sources(input_text, access_other_history):
#     global conversation_history1
#     global conversation_history2

#     if access_other_history:
#         conversation_history2.extend(conversation_history1)
#         conversation_history1.clear()

#     return process_chatbot_with_sources(input_text, conversation_history2)

# def process_chatbot_with_sources(input_text, conversation_history):
#     conversation_history.append(f"User: {input_text}\n")

#     input_with_history = "".join(conversation_history)

#     if len(input_with_history) > max_input_size:
#         response_text = "The input size, including conversation history, exceeds the maximum allowed input size. Please reduce the input or conversation history."
#         conversation_history.append(f"AI: {response_text}\n")
#         return response_text

#     index = VectorStoreIndexWrapper.load_from_disk('index.json')
#     response = index.query_with_sources(input_with_history)

#     formatted_response = f"{response.response}\n\nSources:\n"
#     for source in response.sources:
#         formatted_response += f"{source}\n"

#     conversation_history.append(f"AI: {formatted_response}\n")

#     return formatted_response

# iface1 = gr.Interface(
#     fn=chatbot1,
#     inputs=[
#         gr.components.Textbox(lines=15, label="Enter your text (1)", placeholder="Type your message here..."),
#         gr.components.Checkbox(label="Access other conversation history"),
#         gr.components.Checkbox(label="Show sources")  # checkbox component to enable chatbot1_with_sources
#     ],
#     outputs=gr.components.Text(label="text (1)"),
#     title="A GPT-4 AI on Your Custom Files"
# )

# iface2 = gr.Interface(
#     fn=chatbot2,
#     inputs=[
#         gr.components.Textbox(lines=15, label="Enter your text (2)", placeholder="Type your message here..."),
#         gr.components.Checkbox(label="Access other conversation history"),
#         gr.components.Checkbox(label="Show sources")  # checkbox component to enable chatbot2_with_sources
#     ],
#     outputs=gr.components.Text(label="text (2)"),
#     title="A GPT-4 AI on Your Custom Files"
# )

# parallel_iface = gr.Parallel(iface1, iface2)

# index = construct_index("docs")
# parallel_iface.launch(share=True)


# iface1 = gradio.Interface(
#     fn=chatbot1,
#     inputs=[
#         gradio.components.Input(type="textbox", lines=15, label="Enter your text (1)", placeholder="Type your message here..."),
#         gradio.components.Input(type="checkbox", label="Access other conversation history"),
#         gradio.components.Input(type="checkbox", label="Show sources")  # checkbox component to enable chatbot1_with_sources
#     ],
#     outputs=gradio.components.Output(type="text", label="text (1)"),
#     title="A GPT-4 AI on Your Custom Files"
# )

# iface2 = gradio.Interface(
#     fn=chatbot2,
#     inputs=[
#         gradio.components.Input(type="textbox", lines=15, label="Enter your text (2)", placeholder="Type your message here..."),
#         gradio.components.Input(type="checkbox", label="Access other conversation history"),
#         gradio.components.Input(type="checkbox", label="Show sources")  # checkbox component to enable chatbot2_with_sources
#     ],
#     outputs=gradio.components.Output(type="text", label="text (2)"),
#     title="A GPT-4 AI on Your Custom Files"
# )

# parallel_iface = gradio.Parallel(iface1, iface2)

# index = construct_index("docs")
# parallel_iface.launch(share=True)

# iface1 = gr.Interface(
#     fn=chatbot1,
#     inputs=[
#         gr.inputs.Textbox(lines=15, label="Enter your text (1)", placeholder="Type your message here..."),
#         gr.inputs.Checkbox(label="Access other conversation history"),
#         gr.inputs.Checkbox(label="Show sources")  # checkbox component to enable chatbot1_with_sources
#     ],
#     outputs=gr.outputs.Text(label="text (1)"),
#     title="A GPT-4 AI on Your Custom Files"
# )

# iface2 = gr.Interface(
#     fn=chatbot2,
#     inputs=[
#         gr.inputs.Textbox(lines=15, label="Enter your text (2)", placeholder="Type your message here..."),
#         gr.inputs.Checkbox(label="Access other conversation history"),
#         gr.inputs.Checkbox(label="Show sources")  # checkbox component to enable chatbot2_with_sources
#     ],
#     outputs=gr.outputs.Text(label="text (2)"),
#     title="A GPT-4 AI on Your Custom Files"
# )

# parallel_iface = gr.Parallel(iface1, iface2)

# index = construct_index("docs")
# parallel_iface.launch(share=True)

# Issue: Messed up Gradio Interface parallels
# from gradio.mix import Parallel

# parallel_fn1 = Parallel(chatbot1, chatbot1_with_sources)
# parallel_fn2 = Parallel(chatbot2, chatbot2_with_sources)

# iface = gr.Interface(
#     fn=[parallel_fn1, parallel_fn2],
#     inputs=[
#         gr.inputs.Textbox(lines=15, label="Enter your text (1)", placeholder="Type your message here..."),
#         gr.inputs.Textbox(lines=15, label="Enter your text (2)", placeholder="Type your message here..."),
#         gr.inputs.Checkbox(label="Access other conversation history"),
#         gr.inputs.Checkbox(label="Show sources")  # checkbox component to enable query_with_sources
#     ],
#     outputs=[
#         gr.outputs.Text(label="text (1)"),
#         gr.outputs.Text(label="text (2)")
#     ],
#     title="A GPT-4 AI on Your Custom Files"
# )

# index = construct_index("docs")
# iface.launch(share=True)

# Issue: Version 2 old code
# def construct_index(directory_path):
#     loader = TextLoader(directory_path)
#     index = VectorstoreIndexCreator().from_loaders([loader])
#     index.save_to_disk('index.json')
#     return index
#
# def chatbot(input_text, access_other_history=False):
#     global conversation_history
#     current_history = '1' if not access_other_history else '2'

#     conversation_history[current_history].append(f"User: {input_text}\n")
#     input_with_history = "".join(conversation_history[current_history])

#     index = VectorStoreIndexWrapper.load_from_disk('index.json')
#     response = index.query(input_with_history)

#     conversation_history[current_history].append(f"AI: {response}\n")
#     return response
#
# def chatbot_with_sources(input_text, access_other_history=False, show_sources=False):
#     global conversation_history
#     current_history = '1' if not access_other_history else '2'

#     conversation_history[current_history].append(f"User: {input_text}\n")
#     input_with_history = "".join(conversation_history[current_history])

#     index = VectorStoreIndexWrapper.load_from_disk('index.json')
#     response = index.query_with_sources(input_with_history) if show_sources else index.query(input_with_history)

#     conversation_history[current_history].append(f"AI: {response['answer']}\n")
#     return response['answer'], response.get('sources', '')

# iface = gr.Interface(
#     fn=[chatbot, chatbot_with_sources],
#     inputs=[
#         gr.components.Textbox(lines=15, label="Enter your text (1)", placeholder="Type your message here..."),
#         gr.components.Textbox(lines=15, label="Enter your text (2)", placeholder="Type your message here..."),
#         gr.components.Checkbox(label="Access other conversation history"),
#         gr.components.Checkbox(label="Show sources")
#     ],
#     outputs=[
#         gr.components.Text(label="text (1)"),
#         gr.components.Text(label="sources (1)"),
#         gr.components.Text(label="text (2)"),
#         gr.components.Text(label="sources (2)")
#     ],
#     title="A GPT-4 AI on Your Custom Files"
# )

# iface = gr.Interface(
#     fn=gr.mix.Parallel(chatbot1, chatbot2),
#     inputs=[
#         gr.components.Textbox(lines=15, label="Enter your text (1)", placeholder="Type your message here..."),
#         gr.components.Textbox(lines=15, label="Enter your text (2)", placeholder="Type your message here..."),
#         gr.components.Checkbox(label="Access other conversation history"),
#     ],
#     outputs=[
#         gr.components.Text(label="text (1)"),
#         gr.components.Text(label="text (2)")
#     ],
#     title="A GPT-4 AI on Your Custom Files"
# )

# def chatbot_with_sources(input_text, access_other_history=False, show_sources=False):
#     global conversation_history
#     current_history = '1' if not access_other_history else '2'

#     conversation_history[current_history].append(f"User: {input_text}\n")
#     input_with_history = "".join(conversation_history[current_history])

#     index = VectorStoreIndexWrapper.load_from_disk('index.json')
#     response = index.query_with_sources(input_with_history) if show_sources else index.query(input_with_history)
