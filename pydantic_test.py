from langchain_ollama import ChatOllama
from pydantic import BaseModel

# --------------------------------------------------
# 1. Define expected output structure
# --------------------------------------------------

class InventoryResult(BaseModel):

    item: str
    current_stock: int
    reorder_level: int
    needs_reorder: bool
    reason: str


# --------------------------------------------------
# 2. Create Gemma
# --------------------------------------------------

llm = ChatOllama(
    model="qwen3:4b"
)


# --------------------------------------------------
# 3. Tell LangChain the expected output structure
# --------------------------------------------------

structured_llm = llm.with_structured_output(
    InventoryResult
)


# --------------------------------------------------
# 4. Ask the model
# --------------------------------------------------

result = structured_llm.invoke(
    """
    Analyze this inventory:

    Item: Blue Shirt
    Current stock: 50
    Reorder level: 100

    Determine whether the item needs to be reordered.
    """
)


# --------------------------------------------------
# 5. Display result
# --------------------------------------------------

print("Item:", result.item)
print("Current Stock:", result.current_stock)
print("Reorder Level:", result.reorder_level)
print("Needs Reorder:", result.needs_reorder)
print("Reason:", result.reason)