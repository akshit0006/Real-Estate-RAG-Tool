from langchain_core.prompts import PromptTemplate

template = """Given the following extracted parts of a long document and a question,
create a final answer with references ("SOURCES").
If you don't know the answer, just say that you don't know.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["summaries", "question"]
)

EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)
