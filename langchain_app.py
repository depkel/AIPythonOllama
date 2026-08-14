#creates a LangChain chat model object configured to communicate with Ollama.
#ChatOllama is the LangChain integration/interface that communicates with Ollama.
#LangChain gives you a common interface what ever be underlying model

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(
    model="gemma3:4b"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an inventory assistant."
    ),
    (
        "user",
        "The item is {item} "
        "Current quantity is {quantity}."
        "Explain whether the inventory level may require attention."
    )
])

chain = prompt | llm

response = chain.invoke({
    "item": "shirt",
     "quantity": 50
})

print(response.content)

#print(messages)
#print(type(response))
#print(response)
#print(response.usage_metadata)
