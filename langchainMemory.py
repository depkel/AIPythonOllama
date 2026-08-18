from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
llm = ChatOllama(
    model="gemma3:4b"
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=
            """You are strictly a Purchase Order Assistant. 
                    Only answer questions related to purchasing,
                    inventory, suppliers and purchase orders.
                    If the question is unrelated,
                    politely explain that you only handle purchasing-related questions.
        
                    Your responsibilities are:
        
                    1. Help users understand inventory.
                    2. Help determine whether items need reordering.
                    3. Help with purchase order calculations.
                    4. Explain suppliers and purchasing concepts.
                    5. Give clear and practical answers. Do not give over explanation. Just like 2 or 3 line summary only.
        
                    Always explain your reasoning in simple business language. .
                    If you don't have enough information, ask the user for the missing information."""
    ),
    MessagesPlaceholder(variable_name="history"),
    (
        "user",
        "{question}"
    )
])

conversation_history = []

print("=" * 50)
print("LangChain PO Assistant")
print("Type 'exit' to quit")
print("=" * 50)

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    messages = prompt.invoke({
        "history": conversation_history,
        "question": question
    })
    #print("\n--- Messages ---")
    #print(messages)

    # print("\n--- HISTORY ---")
    # print(conversation_history)
    # print("---------------") 

    #response = llm.invoke(messages)  # waits for the entire response.
    #print("\nGemma:", response.content)

    print("\nGemma:")
    full_response = ""  
    for chunk in llm.stream(messages):
        # Get text from the current chunk
        content = chunk.content
        # Display immediately
        print(content, end="", flush=True)
        # Build complete response
        full_response += content  # This is particularly important because you're also maintaining conversation history.

    print()


    # conversation_history.append({
    #     "role": "user",
    #     "content": question
    # })

    # conversation_history.append({
    #     "role": "assistant",
    #     "content": response.content
    # })
    conversation_history.append(
        HumanMessage(content = question)
    )
    conversation_history.append(
        AIMessage(content = full_response) # response.content
    )