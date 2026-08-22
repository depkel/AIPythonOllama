from langchain_ollama import ChatOllama

from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# ============================================================
# 1. TOOLS
# ============================================================

@tool
def get_inventory(item_code: str, store_id: str) -> dict:
    """Get the current inventory quantity for an item at a specific store."""

    inventory = {
        ("SHIRT001", "BLR001"): 45
    }

    return {
        "item_code": item_code,
        "store_id": store_id,
        "current_stock": inventory.get(
            (item_code, store_id),
            0
        )
    }


@tool
def get_open_purchase_orders(
    item_code: str,
    store_id: str
) -> dict:
    """Get the total quantity of open purchase orders for an item at a specific store."""

    open_orders = {
        ("SHIRT001", "BLR001"): 1
    }

    return {
        "item_code": item_code,
        "store_id": store_id,
        "open_order_quantity": open_orders.get(
            (item_code, store_id),
            0
        )
    }


@tool
def get_supplier(
    item_code: str,
    store_id: str
) -> dict:
    """Get the supplier for an item at a specific store."""

    suppliers = {
        ("SHIRT001", "BLR001"): "ABC Textiles"
    }

    return {
        "item_code": item_code,
        "store_id": store_id,
        "supplier": suppliers.get(
            (item_code, store_id),
            "Unknown"
        )
    }


@tool
def get_reorder_policy(
    item_code: str,
    store_id: str
) -> dict:
    """Get the reorder quantity policy for an item at a specific store."""

    policies = {
        ("SHIRT001", "BLR001"): 10
    }

    return {
        "item_code": item_code,
        "store_id": store_id,
        "reorder_quantity": policies.get(
            (item_code, store_id),
            0
        )
    }


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
# 5. CREATE TOOL LOOKUP
# ============================================================

tool_map = {
    tool.name: tool
    for tool in tools
}


# ============================================================
# 6. USER QUESTION
# ============================================================

user_question = """
For SHIRT001 in store BLR001,
give me the inventory, incoming quantity,
supplier and reorder quantity.
"""


# ============================================================
# 7. INITIAL MESSAGE
# ============================================================

messages = [
    {
        "role": "user",
        "content": user_question
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
    print(f"AGENT ITERATION: {iteration + 1} Calling LLM...")
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

        print("\nFINAL ANSWER:")
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

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]


        print("\nExecuting tool:")
        print(f"  Name      : {tool_name}")
        print(f"  Arguments : {tool_args}")


        # ----------------------------------------------------
        # Find actual Python tool
        # ----------------------------------------------------

        tool = tool_map.get(tool_name)


        if tool is None:

            print(
                f"ERROR: Tool '{tool_name}' not found."
            )

            continue


        # ----------------------------------------------------
        # Execute tool
        # ----------------------------------------------------

        try:

            tool_result = tool.invoke(
                tool_args
            )


            print(
                f"  Result    : {tool_result}"
            )


        except Exception as e:

            print(
                f"  Tool Error: {e}"
            )

            tool_result = {
                "error": str(e)
            }


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
    print("MAXIMUM ITERATIONS REACHED")
    print("=" * 60)

    print(
        f"The agent stopped after "
        f"{max_iterations} iterations."
    )