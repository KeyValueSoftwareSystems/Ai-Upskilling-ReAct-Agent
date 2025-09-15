"""
tools.py - Defines tools and data models for the ReAct agent

This file contains:
- Pydantic schemas for Order and Shipment data
- In-memory data tables for orders and shipments
- Tool functions that the agent can call
"""

from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class DateRange(BaseModel):
    """Schema for date range"""

    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")


class Order(BaseModel):
    """Schema for order data"""

    order_id: int = Field(description="Unique order identifier")
    customer_id: int = Field(description="Customer identifier")
    customer_name: str = Field(description="Name of the customer")
    product: str = Field(description="Product ordered")
    quantity: int = Field(description="Quantity ordered")
    product_metadata: Dict[str, Any] = Field(description="Metadata of the product")
    status: str = Field(description="Current order status")
    order_date: str = Field(description="Date when order was placed (YYYY-MM-DD)")
    total_amount: float = Field(description="Total amount for the order")


class Shipment(BaseModel):
    """Schema for shipment data"""

    shipment_id: int = Field(description="Unique shipment identifier")
    order_id: int = Field(description="Associated order ID")
    tracking_number: str = Field(description="Tracking number for the shipment")
    status: str = Field(description="Current shipment status")
    carrier: str = Field(description="Shipping carrier")
    estimated_delivery: str = Field(description="Estimated delivery date (YYYY-MM-DD)")
    current_location: str = Field(description="Current location of the shipment")
    destination: str = Field(description="Destination of the shipment")
    origin: str = Field(description="Origin of the shipment")
    shipping_date: str = Field(
        description="Date when shipment was shipped (YYYY-MM-DD)"
    )
    delayed_reason: str = Field(description="Reason for delayed shipment")


# In-memory data tables (demo-ready for Sept 2025)
ORDERS_TABLE = {
    1234: Order(
        order_id=1234,
        customer_id=1001,
        customer_name="John Doe",
        product="Wireless Headphones",
        quantity=1,
        status="shipped",
        order_date="2025-09-02",
        total_amount=99.99,
        product_metadata={"color": "black", "size": "medium"},
    ),
    1235: Order(
        order_id=1235,
        customer_id=1002,
        customer_name="Jane Smith",
        product="Laptop Stand",
        quantity=2,
        status="processing",
        order_date="2025-09-09",
        total_amount=49.98,
        product_metadata={"color": "white", "size": "large"},
    ),
    1236: Order(
        order_id=1236,
        customer_id=1003,
        customer_name="Bob Johnson",
        product="USB-C Cable",
        quantity=3,
        status="delivered",
        order_date="2025-09-08",
        total_amount=29.97,
        product_metadata={"color": "black", "size": "small"},
    ),
    1237: Order(
        order_id=1237,
        customer_id=1001,
        customer_name="John Doe",
        product="Smart Watch",
        quantity=1,
        status="pending",
        order_date="2025-09-10",
        total_amount=199.99,
        product_metadata={"color": "black", "size": "medium"},
    ),
}

SHIPMENTS_TABLE = {
    1001: Shipment(
        shipment_id=1001,
        order_id=1234,
        tracking_number="TRK123456789",
        status="delayed",
        carrier="FedEx",
        estimated_delivery="2025-09-05",
        current_location="Distribution Center - Chicago",
        destination="New York, NY",
        origin="Los Angeles, CA",
        shipping_date="2025-09-03",
        delayed_reason="Delivery delayed due to weather conditions",
    ),
    1002: Shipment(
        shipment_id=1002,
        order_id=1236,
        tracking_number="TRK987654321",
        status="delivered",
        carrier="UPS",
        estimated_delivery="2025-09-08",
        current_location="Delivered to customer",
        destination="Boston, MA",
        origin="Newark, NJ",
        shipping_date="2025-09-07",
        delayed_reason="in transit",
    ),
}


@tool
def get_order(order_id: int) -> Dict[str, Any]:
    """
    Retrieve order information by order ID

    Args:
        order_id: The order ID to look up

    Returns:
        Dictionary containing order information or error message
    """
    if isinstance(order_id, str):
        order_id = int(order_id)
    if order_id in ORDERS_TABLE:
        order = ORDERS_TABLE[order_id]
        return {
            "success": True,
            "data": order.model_dump(),
            "message": f"Order {order_id} found",
        }
    else:
        return {
            "success": False,
            "data": f"Order {order_id} not found",
            "message": f"Order {order_id} not found",
        }


@tool
def get_order_by_customer_id(
    customer_id: int, date_range: DateRange | None = None
) -> Dict[str, Any]:
    """
    Retrieve order information by customer ID and optionally filter by date range

    Args:
        customer_id: The customer ID to look up
        date_range: The date range to filter by (optional) in the format {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}
    Returns:
        Dictionary containing order information or error message
    """
    if isinstance(customer_id, str):
        customer_id = int(customer_id)
    orders = []
    for order in ORDERS_TABLE.values():
        if order.customer_id == customer_id:
            if date_range:
                if (
                    order.order_date >= date_range.start_date
                    and order.order_date <= date_range.end_date
                ):
                    orders.append(order)
            else:
                orders.append(order)
    if orders:
        return {
            "success": True,
            "data": [order.model_dump() for order in orders],
            "message": f"{len(orders)} orders found for customer {customer_id} {f'in the date range {date_range.start_date} to {date_range.end_date}' if date_range else ''}",
        }
    else:
        return {
            "success": False,
            "data": f"No orders found for customer {customer_id}",
            "message": f"No orders found for customer {customer_id}",
        }


@tool
def get_shipment(shipment_id: int) -> Dict[str, Any]:
    """
    Retrieve shipment information by shipment ID

    Args:
        shipment_id: The shipment ID to look up

    Returns:
        Dictionary containing shipment information or error message
    """
    if isinstance(shipment_id, str):
        shipment_id = int(shipment_id)
    if shipment_id in SHIPMENTS_TABLE:
        shipment = SHIPMENTS_TABLE[shipment_id]
        return {
            "success": True,
            "data": shipment.model_dump(),
            "message": f"Shipment {shipment_id} found",
        }
    else:
        return {
            "success": False,
            "data": f"Shipment {shipment_id} not found",
            "message": f"Shipment {shipment_id} not found",
        }


@tool
def get_shipment_by_order_id(order_id: int) -> Dict[str, Any]:
    """
    Retrieve shipment information by order ID

    Args:
        order_id: The order ID to find associated shipment

    Returns:
        Dictionary containing shipment information or error message
    """
    if isinstance(order_id, str):
        order_id = int(order_id)
    for shipment in SHIPMENTS_TABLE.values():
        if shipment.order_id == order_id:
            return {
                "success": True,
                "data": shipment.model_dump(),
                "message": f"Shipment found for order {order_id}",
            }

    return {
        "success": False,
        "data": f"No shipment found for order {order_id}",
        "message": f"No shipment found for order {order_id}",
    }
