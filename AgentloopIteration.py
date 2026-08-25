from langchain_ollama import ChatOllama

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# ============================================================
# 1. TOOLS
# ============================================================
# region "TOOLS"    
@tool  # Because you used LangChain's @tool, each tool has a .name "get_inventory"
def get_inventory(item_code: str, store_id: str) -> dict:
    """Get the current inventory quantity for an item at a specific store."""

    # Validation layer  invalid -> error 
    if not item_code:

            return {
                "success": False,
                "error": "INVALID_ITEM_CODE",
                "message": "item_code cannot be empty."
            }


    if not store_id:

        return {
            "success": False,
            "error": "INVALID_STORE_ID",
            "message": "store_id cannot be empty."
        }



    inventory = {
        ("SHIRT001", "BLR001"): 45
    }

    key = (item_code, store_id)

    if key not in inventory:  # Business/data lookup  Not found → Error

        return {
            "success": False,
            "error": "ITEM_NOT_FOUND",
            "message": (
                f"Item {item_code} was not found "
                f"for store {store_id}."
            )
        }

    return {
        "success": True,
        "item_code": item_code,
        "store_id": store_id,
        "current_stock": inventory[key]
    }


@tool #Because you used LangChain's @tool, each tool has a .name. "get_open_purchase_orders"
def get_open_purchase_orders(
    item_code: str,
    store_id: str
) -> dict:
    """Get the total quantity of open purchase orders for an item at a specific store."""

    open_orders = {("SHIRT001", "BLR001"): 1}

    return {
        "item_code": item_code,
        "store_id": store_id,
        "open_order_quantity": open_orders.get(
            (item_code, store_id),
            0
        )
    }


@tool #Because you used LangChain's @tool, each tool has a .name. "get_supplier"
def get_supplier(
    item_code: str,
    store_id: str
) -> dict:
    """Get the supplier for an item at a specific store."""

    suppliers = {("SHIRT001", "BLR001"): "ABC Textiles"}

    return {
        "item_code": item_code,
        "store_id": store_id,
        "supplier": suppliers.get(
            (item_code, store_id),
            "Unknown"
        )
    }


@tool #Because you used LangChain's @tool, each tool has a .name. "get_reorder_policy"
def get_reorder_policy(
    item_code: str,
    store_id: str
) -> dict:
    """Get the reorder level and reorder quantity policy."""

    policies = {
        ("SHIRT001", "BLR001"): {
            "reorder_level": 50,
            "reorder_quantity": 10
        }
    }

    key = (item_code, store_id)

    if key not in policies:

        return {
            "success": False,
            "error": "POLICY_NOT_FOUND",
            "message": (
                f"No reorder policy found for "
                f"{item_code} at {store_id}"
            )
        }

    policy = policies[key]

    return {
        "success": True,
        "item_code": item_code,
        "store_id": store_id,
        "reorder_level": policy["reorder_level"],
        "reorder_quantity": policy["reorder_quantity"]
    }

# deterministic validator is always required 
@tool
def validate_reorder_quantity(
    item_code: str,
    recommended_quantity: int
) -> dict:
    """Validate a recommended purchase order quantity."""

    if recommended_quantity < 0:

        return {
            "success": False,
            "valid": False,
            "error": "NEGATIVE_QUANTITY",
            "message": "Reorder quantity cannot be negative."
        }


    if recommended_quantity > 1000:

        return {
            "success": False,
            "valid": False,
            "error": "QUANTITY_TOO_LARGE",
            "message": "Reorder quantity exceeds the allowed limit."
        }


    return {
        "success": True,
        "valid": True,
        "item_code": item_code,
        "recommended_quantity": recommended_quantity
    }

def calculate_reorder_decision(
    current_stock: int,
    open_order_quantity: int,
    reorder_level: int,
    reorder_quantity: int
) -> dict:

    available_stock = (
        current_stock +
        open_order_quantity
    )

    needs_reorder = (
        available_stock < reorder_level
    )

    if needs_reorder:

        recommended_quantity = reorder_quantity

    else:

        recommended_quantity = 0

    return {
        "available_stock": available_stock,
        "needs_reorder": needs_reorder,
        "recommended_quantity": recommended_quantity
    }

#endregion
# ============================================================
# 2. REGISTER TOOLS
# ============================================================

tools = [
    get_inventory,
    get_open_purchase_orders,
    get_supplier,
    get_reorder_policy
]


# ============================================================
# 3. CREATE LLM
# ============================================================

llm = ChatOllama(
    model="qwen3:4b"
)


# ============================================================
# 4. BIND TOOLS TO LLM
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 5. CREATE TOOL LOOKUP It is a dictionary comprehension.
# ============================================================

tool_map = {
    tool.name: tool
    for tool in tools
}

# result:
# tool_map = {
#     "get_inventory": get_inventory,
#     "get_open_purchase_orders": get_open_purchase_orders,
#     "get_supplier": get_supplier,
#     "get_reorder_policy": get_reorder_policy
# }


# ============================================================
# 6. USER QUESTION
# ============================================================

user_question = """
For SHIRT001 in store BLR001,
give me the inventory, incoming quantity,
supplier and reorder quantity.
"""
If_Wrong_Item  = """
For SHIRT999 in store BLR001,
give me the inventory, incoming quantity,
supplier and reorder quantity.
"""
If_Wrong_Store  = """
For SHIRT001 in store BLR999,
give me the inventory, incoming quantity,
supplier and reorder quantity.
"""

# ============================================================
# 7. INITIAL MESSAGE
# ============================================================

messages = [
    {
        "role": "user",
        "content": If_Wrong_Store #user_question
    }
]


# ============================================================
# 8. MAXIMUM AGENT ITERATIONS
# ============================================================

max_iterations = 10


# ============================================================
# 9. AGENT LOOP
# ============================================================

for iteration in range(max_iterations):

    print("\n")
    print("=" * 60)
    print(f"Agent Iteration: {iteration + 1} Calling LLM...")
    print("=" * 60)


    # --------------------------------------------------------
    # Ask LLM
    # --------------------------------------------------------

    response = llm_with_tools.invoke(messages)
    print('Response from LLM')

    # --------------------------------------------------------
    # Check whether LLM wants to call tools
    # --------------------------------------------------------

    if not response.tool_calls:

        print("\nNo more tool calls.")

        print("\nFinal Answer:")
        print(response.content)

        break


    # --------------------------------------------------------
    # LLM requested one or more tools
    # --------------------------------------------------------

    print("\nLLM requested tools:")

    for tool_call in response.tool_calls:

        print(
            f"  - {tool_call['name']}"
        )


    # --------------------------------------------------------
    # IMPORTANT:
    # Add the LLM response containing tool calls
    # to the conversation history
    # --------------------------------------------------------

    messages.append(response)


    # --------------------------------------------------------
    # Execute each requested tool
    # --------------------------------------------------------

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]  # Your Python program needs to convert that string "get_inventory" into the actual function.

        tool_args = tool_call["args"]


        print("\nExecuting tool:")
        print(f"  Name      : {tool_name}")
        print(f"  Arguments : {tool_args}")


        # ----------------------------------------------------
        # Find actual Python tool
        # ----------------------------------------------------

        tool = tool_map.get(tool_name) # fetch actual function from tool map using "tool name"

        if tool is None:   # A. Unknown tool

                tool_result = {
                    "success": False,
                    "error": "UNKNOWN_TOOL",
                    "message": (
                        f"Tool '{tool_name}' does not exist."
                    )
                }

        else:


        # ----------------------------------------------------
        # Execute tool
        # ----------------------------------------------------

            try:

                tool_result = tool.invoke(tool_args)

            except Exception as e:   # Tool execution error if database is down or API is not available 

                tool_result = {
                "success": False,
                "error": "TOOL_EXECUTION_ERROR",
                "message": str(e)
                }

        print("Tool result:")
        print(tool_result)   
        # ----------------------------------------------------
        # Send tool result back to LLM
        # ----------------------------------------------------

        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
        )


# ============================================================
# 10. MAX ITERATION REACHED
# ============================================================

else:

    print("\n")
    print("=" * 60)
    print("Maximum Iterations Reached")
    print("=" * 60)

    print(
        f"The agent stopped after "
        f"{max_iterations} iterations."
    )


result = calculate_reorder_decision(
    current_stock=45,
    open_order_quantity=1,
    reorder_level=50,
    reorder_quantity=10
)

print(result)