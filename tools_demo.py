from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

# --------------------------------------------------
# 1. Define Tools
# --------------------------------------------------
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

#endregion

# Display tool information

# print("Tool name:")
# print(get_inventory.name)

# print("\nTool description:")
# print(get_inventory.description)

# print("\nTool schema:")
# print(get_inventory.args_schema)

llm = ChatOllama(
    model="qwen3:4b"
)

# We're telling the model: "You have access to this tool."
tools=[get_inventory,get_open_purchase_orders,get_supplier,get_reorder_policy];

llm_with_tools = llm.bind_tools(tools)

q1= "For SHIRT001 in store BLR001, what is the current inventory?"
q2= "For SHIRT001 in store BLR001, tell me the current inventory and supplier?"
q3="For SHIRT001 in BLR001, give me the inventory, incoming quantity, supplier and reorder quantity."

response = llm_with_tools.invoke(
   q3
)
print("\n Qwen response:")
print(response.content)


print("\nTool calls:")
print(response.tool_calls)

# --------------------------------------------------
# 7. Create tool lookup dictionary
# --------------------------------------------------

tool_map = {
    tool.name: tool
    for tool in tools
}
# --------------------------------------------------
# 8. Execute requested tools
# --------------------------------------------------

tool_messages = []

for tool_call in response.tool_calls:

    tool_name = tool_call["name"]

    tool_args = tool_call["args"]

    print("\nExecuting tool:")
    print(tool_name)

    print("Arguments:")
    print(tool_args)

    tool = tool_map[tool_name]

    tool_result = tool.invoke(
        tool_args
    )

    print("Tool result:")
    print(tool_result)


# --------------------------------------------------
# 9. Send result back to Qwen
# --------------------------------------------------

    tool_messages.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    )

# --------------------------------------------------
# 10. Ask Qwen again with tool result
# --------------------------------------------------

messages = [
    {
        "role": "user",
        "content": q3
    },
    response,
    *tool_messages
]

final_response = llm_with_tools.invoke(
    messages
)

# --------------------------------------------------
# 11. Display final answer
# --------------------------------------------------

print("\nFinal answer:")
print(final_response.content)

# region manual tool invoking  
# # Manually invoke the tool
# result = get_inventory.invoke({
#     "item_code": "SHIRT001",
#     "store_id":"BLR001"
# })

# print("\nInventory Tool result:")
# print(result)

# result2 = get_open_purchase_orders.invoke({
#     "item_code": "SHIRT001",
#     "store_id":"BLR001"
# })

# print("\n get_open_purchase_orders Tool result:")
# print(result2)

# result3 = get_supplier.invoke({
#     "item_code": "SHIRT001",
#     "store_id":"BLR001"
# })

# print("\n get_supplier Tool result:")
# print(result3)

# result4 = get_reorder_policy.invoke({
#     "item_code": "SHIRT001",
#     "store_id":"BLR001"
# })

# print("\n get_reorder_policy Tool result:")
# print(result4)
# endregion