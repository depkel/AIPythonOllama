from ollama import chat

print("=" * 50)
print("Purchase Order AI Assistant")
print("Type 'exit' to quit")
print("=" * 50)

conversation_history = [

    {
        "role": "system",
        "content": """
            You are strictly a Purchase Order Assistant.
            Only answer questions related to purchasing,
            inventory, suppliers and purchase orders.
            If the question is unrelated,
            politely explain that you only handle purchasing-related questions.

            Your responsibilities are:

            1. Help users understand inventory.
            2. Help determine whether items need reordering.
            3. Help with purchase order calculations.
            4. Explain suppliers and purchasing concepts.
            5. Give clear and practical answers.

            Always explain your reasoning in simple business language.
            If you don't have enough information, ask the user for the missing information.
            """
    }

]

while True:

    user_input = input("\nYou : ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    conversation_history.append({
        "role": "user",
        "content": user_input
    })

    response = chat(
        model="gemma3:4b",
        messages=conversation_history
    )

    assistant_response = response["message"]["content"]

    print("\nGemma :")
    print(assistant_response)

    conversation_history.append({
        "role": "assistant",
        "content": assistant_response
    })