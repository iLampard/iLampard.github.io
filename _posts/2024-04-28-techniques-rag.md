---
layout: distill
title: Basic and Advanced Techniques on RAG
date: 2024-04-28 00:20:00
description: detailed techniques on RAG, with LangChain code examples
tags: RAG code 
categories: LLM
featured: true

# toc:
#  beginning: true

toc:
  - name: Introduction to RAG
    subsections:
    - name: Indexing
    - name: Embedding Model
    - name: Retriever
  - name: Advanced Techniques in Retrievers
    subsections:
    - name: Query Rewriting
    - name: RAG Fusion
    - name: Step Back Prompting
    - name: Corrective RAG
    - name: RAFT
  - name: Evaluation
  - name: Reference


 
# chart:
  # vega_lite: true


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

# Make splits
splits = text_splitter.split_documents(blog_docs)

print(splits[0])
> page_content='LLM Powered Autonomous Agents' metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}

print(splits[1])
> page_content='Date: June 23, 2023  |  Estimated Reading Time: 31 min  |  Author: Lilian Weng' metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}

print(len(splits))
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
retriever = vectorstore.as_retriever()

docs = retriever.get_relevant_documents("What is task decomposition for LLM agents?")

print(docs)
> [Document(page_content='Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'})]
```

## Advanced Techniques in Retrievers

### Query Rewriting

The input query can be ambiguous, causing an inevitab gap between the input text and the knowledge that is really needed to query.

A straightforward way is to use LLM to generate queries from multiple perspective, a.k.a. multi-query transform. 

The whole process: for a given user input query, it uses an LLM to generate multiple queries from different perspectives. For each query, it retrieves a set of relevant documents and takes the unique union across all queries to get a larger set of potentially relevant documents. Lastly, it feeds all the retrieved documents and let LLM to generate the answer.

```python
from langchain.prompts import ChatPromptTemplate

# Multi Query - generate queries from different perspectives using a prompt
template = """You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import os
os.environ['OPENAI_API_KEY'] = 'xxx'

generate_queries = (
    prompt_perspectives
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)
```

By applying the multi-query approach, we retrieve more documents than the standard approach and run the retrieval process to get an anaswer.

```python
from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve process
question = "What is task decomposition for LLM agents?"

# pass each query to the retriever and remove the duplicate docs
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# retrieve
docs = retrieval_chain.invoke({"question":question})
len(docs)
> 6
```



```python
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0)

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})
> Task decomposition for LLM agents involves parsing user requests into multiple tasks, with the LLM acting as the brain to organize and manage these tasks.
```


### RAG Fusion

How it works
- Performs multi query transformation by translating the user’s queries into similar yet distinct through LLM. (same as MultiQueryRetriever)
- Initialize the vector searches for the original query and its generated similar queries, multiple query generation. (same as MultiQueryRetriever)
- Combine and refine all the query results using $RRF=\frac{1}{rank+k}$, where  $rank$ is the current rank of the documents sorted by distance, and $k$ is a constant smoothing factor that determines the weight given to the existing ranks.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://miro.medium.com/v2/resize:fit:1344/format:webp/1*acPUjXj6kIeJHxV5Fgjf9g.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image Source: <a href="https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1">Adrian H. Raudaschl</a> 
</div>


```python
from langchain.prompts import ChatPromptTemplate

# RAG-Fusion: Related
template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)
```



```python
from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
len(docs)
> 6
```

```python
from langchain_core.runnables import RunnablePassthrough

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})
> Task decomposition for LLM agents involves breaking down large tasks into smaller, manageable subgoals.
```


### Step Back Prompting

The work of `Step Back Prompting` explores
how LLMs can tackle complex tasks involving many low-level details through a two-step process of abstraction-and-reasoning. 
- The first step is to show LLMs how to step back through in-context
learning – prompting them to derive high-level abstractions such as concepts and principles for
a specific example. 
- The second step is to leverage the reasoning ability to reason on top of the
high-level concepts and principles.


The details can be found the paper [TAKE A STEP BACK](https://arxiv.org/pdf/2310.06117).


In the following code, we setup a prompting chain to do the abstraction first.
```python
# Few Shot Examples to show how to abstract the question
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?", # abstraction
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?", # abstraction
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)
```

Then we can see how it abstract the question "What is task decomposition for LLM agents?".

```python
generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
question = "What is task decomposition for LLM agents?"
generate_queries_step_back.invoke({"question": question})
> What is task decomposition in general?
```


```python

from langchain_core.runnables import RunnableLambda

# Response prompt 
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

chain.invoke({"question": question})

> Task decomposition for LLM agents refers to the process of breaking down complex tasks into smaller, more manageable subtasks that can be easily understood and executed by the large language model (LLM). This decomposition allows the LLM to effectively parse user requests and plan out the necessary steps to accomplish the overall task.

> In the context of LLM-powered autonomous agent systems, task decomposition is a crucial component of the overall workflow...

> Additionally, task decomposition with LLM involves simple prompting techniques, such as providing specific steps for a task (e.g., "Steps for XYZ. 1.") to guide the model in understanding and executing the task effectively...

> Overall, task decomposition for LLM agents plays a crucial role in enhancing the performance and capabilities of autonomous agent systems by enabling effective task planning, model selection, and task execution...
```


### Corrective RAG 

[Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) (CRAG) is proposed to enhance the robustness of generation when errors in retrieval are introduced.

Part of CRAG, is a `lightweight trainable retrieval evaluator` which assesses the overall quality of retrieved documents, providing a confidence degree to trigger different knowledge retrieval actions.
- The retrieval evaluator quantifies a confidence degree, enabling different knowledge retrieval actions such as `Correct`, `Incorrect`, `Ambiguous` based on the assessment.
- For `Incorrect` and `Ambiguous` cases, large-scale `web searches` are integrated strategically to address limitations in static and limited corpora, aiming to provide a broader and more diverse set of information.


### Retrieval Augmented FineTuning (RAFT)

Different from classical approaches, [RAFT](https://arxiv.org/pdf/2403.10131) combines RAG and Supervised FineTuning (SFT) at both train and test time.

Consider the supervised fine-tuning (SFT) setting for a Question-Answer dataset. The formulation consists of the Dataset (D) from which a set of Question (Q) and corresponding answer (A) pairs are derived or already available.
- Train: $Q \rightarrow A$.
- 0-shot Inference: $Q \rightarrow A$.
- RAG Inference: $Q + D \rightarrow A$.

In RAFT, we prepare the training data such that each data point contains a question ($Q$), a set of documents ($D_k$), and a corresponding Chain-of-though style answer ($A_*$) generated from one of the document ($D_*$), a.k.a., oracle document while the non-relevant documents are ‘distractors’ ($D_i$).
- Train:
  - $P \% $ of data: $Q + D_* + D_2 + . . . + D_k \rightarrow A_*$
  - $(1 − P) \% $ of data: $Q + D_1 + D_2 + . . . + D_k \rightarrow A_{*}$
- RAG Inference: $Q + D \rightarrow A$.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="https://gorilla.cs.berkeley.edu/assets/img/blog_post_9_RAFT.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image Source: <a href="https://gorilla.cs.berkeley.edu/blogs/9_raft.html">RAFT Github Repo</a> 
</div>





## Evaluation 

Langchain provides various types of RAG eval that users of typically interested in:
- Response compared against reference answer: metrics like correctness measure "how similar/correct is the answer, relative to a ground-truth label"
- Response compared against retrieved docs: metrics like faithfulness, hallucinations, etc. measure "to what extent does the generated response agree with the retrieved context"
- Retrieved docs compared against input: metrics like score @ k, mean reciprocal rank, NDCG, etc. measure "how good are my retrieved results for this query"
  

| Target | Explain | Metrics |
| :----------- | :------------: | ------------: |
|  Answer correctness |    how similar/correct is the answer, relative to a ground-truth label    |      accuracy-ratio |
|  Answer faithfulness |    to what extent does the generated response agree with the retrieved context    |  faithfulness/hallucination ratio |
|  Retro relevance |    how good are the retrieved results for this query    |  score@k, mean reciprocal rank |



The full code snippet can be found at [LangSmith-RAG-Eval](https://docs.smith.langchain.com/cookbook/testing-examples/rag_eval).




## Reference 

- [LangChain - rag from scrach](https://github.com/langchain-ai/rag-from-scratch/tree/main)
- [RAG-Fusion](https://github.com/Raudaschl/rag-fusion/blob/master/main.py)
- [CRAG](https://arxiv.org/abs/2401.15884)



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
