from ollama import chat

msg=input('Enter your prompt:');
response = chat(
    model="gemma3:4b",
    messages=[
        {
            "role": "user",
            "content": msg
        }
    ]
)

print(response["message"]["content"])