---
title: "Conversational RAG agent: create your own ChatGPT"
categories:
  - blog
tags:
  - AI
  - LLM
  - Chatbot
---

<!-- {%- include mathjax.html -%} -->

> Code available [here](https://colab.research.google.com/drive/1dDKQdFWH0_XuwxTmJuUP_2JIHqDKTiyP?usp=sharing)

Have you always desired to create your own chatbot? Then I will show you how.

We will not cover how to create the UI here, but I will show you how to choose a LLM, how to add a RAG to it, and how to let this LLM remember the full history of the conversation you're having with it.

Let's go.

## Choose model

First of all, we need a model:

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
)

chat_model = ChatHuggingFace(llm=llm)
```

## RAG

Then, we need to create a RAG:

```python
import os, sys
import typing as ty

from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader

loader = DirectoryLoader(
    path=os.path.join('.'),
    glob="*.pdf",
    recursive=True,
)

docs: ty.List[Document] = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
chunked_docs = splitter.split_documents(docs)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

# For all model names, see: https://www.sbert.net/docs/pretrained_models.html
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
db = FAISS.from_documents(chunked_docs, embedding=embedding)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
```

Now we add the history of the conversation to the RAG:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

## Agent

Let's give this retriever as a tool to the agent that we'll build:

```python
from langchain.tools.retriever import create_retriever_tool

# Build retriever tool
tool = create_retriever_tool(
    history_aware_retriever,
    name="document_retriever",
    description="Searches and returns excerpts from the local database of documents.",
)
tools = [tool]
```

And here the agent:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent_executor = create_react_agent(chat_model, tools, checkpointer=memory)
```

## Evaluate

Now, let's ask it a question:

```python
config = {"configurable": {"thread_id": "abc123"}}

for event in agent_executor.stream(
    {"messages": [HumanMessage(content="What is Task Decomposition?")]},
    config=config,
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

The response I got was:

```text
================================ Human Message =================================

What is Task Decomposition?
================================== Ai Message ==================================

Task decomposition is a process in which a complex task is broken down into smaller, more manageable subtasks.
This approach allows for more efficient and effective completion of the overall task by making it easier to understand, plan, and execute each individual subtask.
Task decomposition is commonly used in project management, engineering, and other fields where complex tasks need to be completed within specific timeframes and resource constraints.
By breaking down tasks into smaller components, it becomes easier to identify potential issues, allocate resources, and monitor progress.
Additionally, task decomposition can help to ensure that all necessary steps are taken to complete the task successfully.
```

Now let's ask a second question that references the first one, and let's see if the agent remembers it:

```python
query = "What are common ways of doing it?"

for event in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]},
    config=config,
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

The response was:

```text
================================ Human Message =================================

What are common ways of doing it?
================================== Ai Message ==================================

There are several common ways to perform task decomposition:

1. Top-down approach: This is a hierarchical method where the overall task is broken down into smaller subtasks, which are then further decomposed into even smaller subtasks until the desired level of detail is reached. This approach is useful when the overall task is well-defined and has clear deliverables.

2. Bottom-up approach: This is a reverse engineering method where the smaller subtasks are identified first, and then they are grouped together to form larger subtasks and eventually the overall task. This approach is useful when the smaller subtasks are more familiar and better understood than the overall task.

3. Iterative approach: This is a cyclical process where the overall task is broken down into subtasks, which are then executed and evaluated. Based on the results, the subtasks may be refined, combined, or split into smaller subtasks until the desired level of detail is reached. This approach is useful when the overall task is complex and requires multiple iterations to refine and optimize.

4. Cross-functional approach: This is a collaborative method where the overall task is broken down into subtasks that involve multiple functional areas or teams. This approach is useful when the overall task requires expertise from multiple disciplines or departments.

5. Agile approach: This is an iterative and incremental method where the overall task is broken down into small, deliverable-based subtasks that are executed in short sprints. This approach is useful when the overall task is highly dynamic and requires frequent adjustments and adaptations.

These approaches can be used individually or in combination, depending on the nature of the overall task and the preferences of the project team.
```

As you can see, this LLM agent remembered the question.

Of course, it can also access any documents you give it. Here, we won't go over that as we already covered this part in a previos post.
