---
layout: distill
title: Basic and Advanced Techniques on RAG
date: 2024-04-28 00:20:00
description: detailed techniques on RAG, with LangChain code examples
tags: RAG code 
categories: LLM
featured: true

# chart:
  # vega_lite: true


toc:
  - name: Introduction to RAG
    subsections:
    - name: Indexing
    - name: Embedding Model
    - name: Retriever
  - name: Reference
---


## Introduction to RAG

Many LLM applications require user-specific data that is not part of the model's training set. The primary way of accomplishing this is through Retrieval Augmented Generation (RAG).
In RAG process, external data is retrieved and then passed to the LLM when doing the generation step. We take [Langchain](https://python.langchain.com/docs/get_started/introduction) as the codebase to understand this process.


RAG typically involves three process: indexing, retrieval and generation.

### Indexing

In indexing process, the systems sync documents from external source into a vector store. Assume we load a website as a document.

```python
! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain langchain_text_splitters sentence_transformers
```


```python
# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

len(blog_docs[0].page_content)
> 43131
```

Once loaded, we need to split the long document into smaller chunks that can fit into the model's context window. LangChain has a number of built-in document transformers that make it easy to split, combine, filter, and otherwise manipulate documents.

Base on LangChain's document, at a high level, text splitters firstly split the text up into small, semantically meaningful chunks (often sentences); then combine these small chunks into a larger chunk until reaching a certain size (as measured by some function). Once reaching that size, the splitter makes that chunk its own piece of text and then start creating a new chunk of text with some overlap (to keep context between chunks).

By default, it is recommended to use RecursiveCharacterTextSplitter for generic text. It is parameterized by a list of characters (The default list is `["\n\n", "\n", " ", ""]`.). It tries to split on them in order until the chunks are small enough. It has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.


```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([blog_docs[0].page_content])
print(texts[0])
> page_content='LLM Powered Autonomous Agents'

print(texts[1])
> page_content='Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  Author: Lilian Weng'

print(len(texts))
> 611
```

### Embedding Model

The embedding model creates a vector representation for a piece of text so we can think about text in the vector space, and do things like semantic search where we look for pieces of text that are most similar in the vector space.

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
```

```python
emb = hf.embed_query("hi this is harrison")
print(len(emb))
> 384
```

We can initialize the vector store object following the following example
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=texts, 
                                    embedding=hf)
```                               


### Retriever

A retriever is an interface that returns documents given an unstructured query.

Retriever vs vectorstore:  
- A retriever is more general than a vector store. A retriever does not need to be able to store documents, only to return (or retrieve) them. 
- Vector stores can be used as the backbone of a retriever, but there are other types of retrievers as well.


```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = retriever.get_relevant_documents("What is title?")
print(docs)
> [Document(page_content='"content": "Summary of areas that need clarification:\\n1. Specifics of the Super Mario game')]
```



## Reference 

- [LangChain - rag from scrach](https://sebastianraschka.com/blog/2023/llm-mixed-precision-copy.html](https://github.com/langchain-ai/rag-from-scratch/tree/main)



<!--
````markdown
```vega_lite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "description": "A dot plot showing each movie in the database, and the difference from the average movie rating. The display is sorted by year to visualize everything in sequential order. The graph is for all Movies before 2019.",
  "data": {
    "url": "https://raw.githubusercontent.com/vega/vega/main/docs/data/movies.json"
  },
  "transform": [
    {"filter": "datum['IMDB Rating'] != null"},
    {"filter": {"timeUnit": "year", "field": "Release Date", "range": [null, 2019]}},
    {
      "joinaggregate": [{
        "op": "mean",
        "field": "IMDB Rating",
        "as": "AverageRating"
      }]
    },
    {
      "calculate": "datum['IMDB Rating'] - datum.AverageRating",
      "as": "RatingDelta"
    }
  ],
  "mark": "point",
  "encoding": {
    "x": {
      "field": "Release Date",
      "type": "temporal"
    },
    "y": {
      "field": "RatingDelta",
      "type": "quantitative",
      "title": "Rating Delta"
    },
    "color": {
      "field": "RatingDelta",
      "type": "quantitative",
      "scale": {"domainMid": 0},
      "title": "Rating Delta"
    }
  }
}
```
````

Which generates:

```vega_lite
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "description": "A dot plot showing each movie in the database, and the difference from the average movie rating. The display is sorted by year to visualize everything in sequential order. The graph is for all Movies before 2019.",
  "data": {
    "url": "https://raw.githubusercontent.com/vega/vega/main/docs/data/movies.json"
  },
  "transform": [
    {"filter": "datum['IMDB Rating'] != null"},
    {"filter": {"timeUnit": "year", "field": "Release Date", "range": [null, 2019]}},
    {
      "joinaggregate": [{
        "op": "mean",
        "field": "IMDB Rating",
        "as": "AverageRating"
      }]
    },
    {
      "calculate": "datum['IMDB Rating'] - datum.AverageRating",
      "as": "RatingDelta"
    }
  ],
  "mark": "point",
  "encoding": {
    "x": {
      "field": "Release Date",
      "type": "temporal"
    },
    "y": {
      "field": "RatingDelta",
      "type": "quantitative",
      "title": "Rating Delta"
    },
    "color": {
      "field": "RatingDelta",
      "type": "quantitative",
      "scale": {"domainMid": 0},
      "title": "Rating Delta"
    }
  }
}
```

This plot supports both light and dark themes.


--->
