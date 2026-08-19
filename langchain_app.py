#creates a LangChain chat model object configured to communicate with Ollama.
#ChatOllama is the LangChain integration/interface that communicates with Ollama.
#LangChain gives you a common interface what ever be underlying model

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="qwen3:4b"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a Purchase Order Assistant."
    ),
    (
        "user",
        "Item: {item}"
        "Current stock: {stock}"
        "Supplier: {supplier}"
        "Explain the purchasing situation."
    )
])

chain = prompt | llm

response = chain.invoke({
    "item": "Blue Shirt",
    "stock": 40,
    "supplier": "ABC Textiles"
})

print(response.content)

#print(messages)
#print(type(response))
#print(response)
#print(response.usage_metadata)
