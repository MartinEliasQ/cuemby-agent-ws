"""
Restaurant Service Workflow
----------------------------
This module implements a LangGraph workflow for handling restaurant-related
customer interactions, classifying them as either order requests or questions,
and processing them accordingly.
"""

import os
import logging
from enum import Enum
from typing import List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Configure logging
logging.basicConfig(
    level=logging.INFO, format=" %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("restaurant_service")

# Configure OpenAI API
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
llm = ChatOpenAI(model="gpt-4o-2024-08-06")


# Define data models
class PaymentMethod(str, Enum):
    """Supported payment methods for orders."""

    CASH = "cash"
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"


class OrderItem(BaseModel):
    """Represents an individual item in an order."""

    name: str
    quantity: int
    notes: Optional[str] = None
    unit_price: Optional[float] = None


class OrderRequest(BaseModel):
    """Represents a complete order request from a customer."""

    order_id: Optional[str] = None
    restaurant: str
    request_date: str
    delivery_address: str
    items: List[OrderItem]
    contact: str
    payment_method: PaymentMethod = PaymentMethod.CASH
    express_delivery: bool
    additional_notes: Optional[str] = None


class State(TypedDict):
    """State management for the workflow."""

    message_type: str
    extraction: dict
    user_response: str
    message_content: str


# Prompts
CLASSIFIER_SYSTEM_PROMPT = """
You are a classifier system. Your task is to classify the message as either ORDER or QUESTION.
An ORDER is a message that contains a delivery order request.
A QUESTION is a message that contains a question about the restaurant.
Only respond with: ORDER, QUESTION
"""

QUESTION_SYSTEM_PROMPT = """
You are an assistant system. Your task is to answer questions about the restaurant.
Please provide a short answer to the question. 
Inform the customer that they will be attended by technical support shortly.
Respond in the context of the customer's question.
"""


# Node functions
def classification_node(state: State) -> State:
    """
    Classifies incoming messages as either ORDER or QUESTION.

    Args:
        state: Current workflow state

    Returns:
        Updated state with message classification
    """
    current_message = state["message_content"]
    human_message = HumanMessage(content=current_message)
    system_message = SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT)

    logger.info("Classifying customer message")
    message_type = llm.invoke([system_message, human_message])

    # Store the classification result
    state["message_type"] = message_type.content.upper()
    logger.info(f"Message classified as: {state['message_type']}")

    return state


def order_processor_node(state: State) -> State:
    """
    Processes order requests by extracting structured information.

    Args:
        state: Current workflow state

    Returns:
        Updated state with extracted order information
    """
    current_message = state["message_content"]
    logger.info("Processing order request")

    # Use structured output to extract order details
    llm_with_structured = llm.with_structured_output(OrderRequest)
    extracted_order = llm_with_structured.invoke(current_message)

    # Store the structured order information
    state["extraction"] = extracted_order.model_dump()
    logger.info(f"Order extracted successfully: {state['extraction']['order_id']}")

    return state


def question_processor_node(state: State) -> State:
    """
    Processes questions about the restaurant by generating helpful responses.

    Args:
        state: Current workflow state

    Returns:
        Updated state with response to customer question
    """
    current_message = state["message_content"]
    logger.info("Processing customer question")

    system_message = SystemMessage(content=QUESTION_SYSTEM_PROMPT)
    human_message = HumanMessage(content=current_message)

    response = llm.invoke([system_message, human_message])

    # Store the generated response
    state["user_response"] = response.content
    logger.info("Generated response to customer question")

    return state


def message_type_router(state: State) -> str:
    """
    Routes the workflow based on the message classification.

    Args:
        state: Current workflow state

    Returns:
        Name of the next node to execute
    """
    if state["message_type"] == "ORDER":
        logger.info("Routing to order processor")
        return "order_processor_node"
    elif state["message_type"] == "QUESTION":
        logger.info("Routing to question processor")
        return "question_processor_node"
    else:
        logger.warning(f"Unknown message type: {state['message_type']}")
        return "question_processor_node"  # Default fallback


# Build workflow graph
def build_workflow():
    """
    Builds and returns the compiled workflow graph.

    Returns:
        Compiled workflow
    """
    logger.info("Building restaurant service workflow")

    # Initialize the graph
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("classification_node", classification_node)
    graph.add_node("order_processor_node", order_processor_node)
    graph.add_node("question_processor_node", question_processor_node)

    # Add edges
    graph.add_edge(START, "classification_node")
    graph.add_conditional_edges("classification_node", message_type_router)
    graph.add_edge("order_processor_node", END)
    graph.add_edge("question_processor_node", END)

    # Compile the workflow
    workflow = graph.compile()
    logger.info("Workflow compiled successfully")

    return workflow


# # Create the workflow
# restaurant_service = build_workflow()


# # Example usage
# if __name__ == "__main__":
#     # Example order
#     order_example = """
# Buenas tardes, quisiera hacer un pedido del restaurante 'El Sabor Casero' para entrega a domicilio. Mi dirección es Calle Robles #42, Apartamento 3B, Colonia Centro. El pedido consiste en 2 hamburguesas clásicas, 1 orden de papas fritas grandes, 1 ensalada César y 2 refrescos de cola medianos. Mi número de contacto es 555-123-4567. Prefiero pagar con tarjeta de crédito al momento de la entrega. ¿Podrían entregarlo lo antes posible? Gracias.
#     """

#     # Example question
#     question_example = """
#     What are your opening hours on weekends? Do you offer vegetarian options?
#     """

#     # Test with an order
#     logger.info("Testing workflow with an order")
#     # order_result = restaurant_service.invoke({"message_content": order_example})

#     # # Test with a question
#     # logger.info("Testing workflow with a question")
#     question_result = restaurant_service.invoke({"message_content": question_example})

#     # Print results
#     # logger.info(f"Order processing result: {order_result.get('extraction', {})}")
#     logger.info(f"Question processing result: {question_result.get('user_response')}")

PORT = os.environ.get("PORT", 8900)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()


class Message(BaseModel):
    message: str


@app.post("/message/")
async def create_item(message: Message):
    workflow = build_workflow()
    try:
        response = workflow.invoke({"message_content": message.message})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
