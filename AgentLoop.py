from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# ==================================================
# TOOLS
# ==================================================
#region "Tools"
@tool #Python decorator
def get_inventory(item_code: str, store_id: str) -> dict:
    """Get current inventory for an item at a specific store."""

    inventory = {
        ("SHIRT001", "BLR001"): 45,
        ("SHIRT002", "BLR001"): 120,
        ("PANT001", "BLR001"): 30
    }

    return {
        "item_code": item_code,
        "store_id": store_id,
        "current_stock": inventory.get((item_code, store_id),0)
    }

@tool
def get_open_purchase_orders(item_code: str, store_id: str)-> dict:
    """Get incoming quantities from open purchase orders."""

    OpenPOs = {
            ("SHIRT001", "BLR001"): 1,
            ("SHIRT002", "BLR001"): 1,
            ("PANT001", "BLR001"): 2
        }

    return {
        "item_code": item_code,
        "store_id": store_id,
        "open_order_quantity": OpenPOs.get((item_code, store_id),0)
    }
    

@tool
def get_supplier(item_code: str,store_id: str) -> dict:
    """Get the primary supplier for an item."""

    ItemSuppliers = {
                ("SHIRT001", "BLR001"): "ABC Textiles",
                ("SHIRT002", "BLR001"): "XYZ Textiles",
                ("PANT001", "BLR001"): "ABC Textiles"
            }
    
    return {
            "item_code": item_code,
            "store_id": store_id,
            "supplier": ItemSuppliers.get((item_code, store_id),"No Supplier found")
        }


@tool
def get_reorder_policy(item_code: str, store_id: str):
    """Get the reorder policy for an item at a store."""

    ItemReorderPolicy = {
                ("SHIRT001", "BLR001"): 10,
                ("SHIRT002", "BLR001"): 15,
                ("PANT001", "BLR001"): 5
            }
    
    return {
            "item_code": item_code,
            "store_id": store_id,
            "reorder_quantity": ItemReorderPolicy.get((item_code, store_id),0)
        }

@tool
def calculate_available_stock(
    current_stock: int,
    open_order_quantity: int
) -> dict:
    """Calculate inventory available after considering incoming purchase orders."""

    available_stock = (
        current_stock +
        open_order_quantity
    )

    return {
        "available_stock": available_stock
    }

#endregion
   

# ==================================================
# TOOL LIST
# ==================================================

tools = [
    get_inventory,
    get_open_purchase_orders,
    get_supplier,
    get_reorder_policy,
    calculate_available_stock
]


# ==================================================
# LLM
# ==================================================

llm = ChatOllama(
    model="qwen3:4b"
)

llm_with_tools = llm.bind_tools(tools)


# ==================================================
# TOOL LOOKUP
# ==================================================

tool_map = {
    tool.name: tool
    for tool in tools
}


# ==================================================
# USER QUESTION
# ==================================================

user_question1 = """
For SHIRT001 in store BLR001,
give me the inventory, incoming quantity,
supplier and reorder quantity.
"""
user_question2="For SHIRT001 in BLR001, determine whether I should reorder."

# ==================================================
# INITIAL MESSAGE
# ==================================================

messages = [
    {
        "role": "user",
        "content": user_question1 
    }
]


# ==================================================
# AGENT LOOP
# ==================================================

while True:

    print("\n================================")
    print("Calling LLM...")
    print("================================")

    response = llm_with_tools.invoke(messages)

    print("Got LLM response")

    # ------------------------------------------------
    # Check whether LLM requested tools
    # ------------------------------------------------

    if response.tool_calls:

        print("\nLLM requested tools:")

        for tool_call in response.tool_calls:

            print(f"- {tool_call['name']}")

        # --------------------------------------------
        # Execute every requested tool
        # --------------------------------------------

        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            tool_args = tool_call["args"]

            print(f"\nExecuting: {tool_name}")

            print(f"Arguments: {tool_args}")

            # Find the Python tool

            tool = tool_map[tool_name]

            # Execute tool

            tool_result = tool.invoke(
                tool_args
            )

            print(f"Result: {tool_result}")

            # ----------------------------------------
            # Add tool result to conversation
            # ----------------------------------------

            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                )
            )


        # --------------------------------------------
        # Continue loop
        # --------------------------------------------

        continue


    # =================================================
    # NO TOOL CALL
    # =================================================

    print("\n================================")
    print("FINAL ANSWER")
    print("================================")

    print(response.content)

    break