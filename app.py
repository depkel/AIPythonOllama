from ollama import chat
msg = input("Enter your question>>")
response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": msg
        }
    ],
    stream=True,
    options={
        "temperature": 1.5
         #"num_predict": 100
    }
)

#print(response["message"]["content"])
for chunk in response:
     print(chunk["message"]["content"], end="", flush=True)

print()