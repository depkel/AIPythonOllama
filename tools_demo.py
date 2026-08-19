from langchain_ollama import ChatOllama
from langchain_core.tools import tool


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
            "Supplier": ItemSuppliers.get((item_code, store_id),"No Supplier found")
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
            "Reorder_Quantity": ItemReorderPolicy.get((item_code, store_id),0)
        }


# Display tool information

# print("Tool name:")
# print(get_inventory.name)

# print("\nTool description:")
# print(get_inventory.description)

# print("\nTool schema:")
# print(get_inventory.args_schema)

llm = ChatOllama(
    model="gemma3:4b"
)

# We're telling the model: "You have access to this tool."
llm_with_tools = llm.bind_tools(  
    [get_inventory,get_open_purchase_orders,get_supplier,get_reorder_policy]
)

# Manually invoke the tool
result = get_inventory.invoke({
    "item_code": "SHIRT001",
    "store_id":"BLR001"
})

print("\nInventory Tool result:")
print(result)

result2 = get_open_purchase_orders.invoke({
    "item_code": "SHIRT001",
    "store_id":"BLR001"
})

print("\n get_open_purchase_orders Tool result:")
print(result2)

result3 = get_supplier.invoke({
    "item_code": "SHIRT001",
    "store_id":"BLR001"
})

print("\n get_supplier Tool result:")
print(result3)

result4 = get_reorder_policy.invoke({
    "item_code": "SHIRT001",
    "store_id":"BLR001"
})

print("\n get_reorder_policy Tool result:")
print(result4)