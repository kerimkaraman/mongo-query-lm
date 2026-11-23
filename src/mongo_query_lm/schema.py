"""The e-commerce schema that synthetic queries are built from."""

from dataclasses import dataclass
from enum import StrEnum


class FieldKind(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class Field:
    """One document field and the values a query may compare it with."""

    name: str
    kind: FieldKind
    choices: tuple[str, ...] = ()
    minimum: float = 0
    maximum: float = 0


CUSTOMER_FIELDS = (
    Field(
        "name",
        FieldKind.STRING,
        choices=("Ayse", "Mehmet", "Elif", "Can", "Zeynep", "Emre"),
    ),
    Field(
        "city",
        FieldKind.STRING,
        choices=("Istanbul", "Ankara", "Izmir", "Bursa", "Antalya"),
    ),
    Field("age", FieldKind.INTEGER, minimum=18, maximum=80),
    Field("signup_year", FieldKind.INTEGER, minimum=2018, maximum=2025),
    Field("is_premium", FieldKind.BOOLEAN),
)

ORDER_FIELDS = (
    Field(
        "status",
        FieldKind.STRING,
        choices=("pending", "shipped", "delivered", "cancelled"),
    ),
    Field("payment_method", FieldKind.STRING, choices=("card", "cash", "transfer")),
    Field("total", FieldKind.DECIMAL, minimum=5, maximum=2000),
    Field("item_count", FieldKind.INTEGER, minimum=1, maximum=10),
    Field("order_year", FieldKind.INTEGER, minimum=2018, maximum=2025),
    Field("is_gift", FieldKind.BOOLEAN),
)

PRODUCT_FIELDS = (
    Field(
        "category",
        FieldKind.STRING,
        choices=("books", "electronics", "toys", "garden", "sports"),
    ),
    Field("brand", FieldKind.STRING, choices=("Acme", "Globex", "Initech", "Umbrella")),
    Field("price", FieldKind.DECIMAL, minimum=1, maximum=1500),
    Field("stock", FieldKind.INTEGER, minimum=0, maximum=500),
    Field("rating", FieldKind.INTEGER, minimum=1, maximum=5),
    Field("is_active", FieldKind.BOOLEAN),
)

SCHEMA: dict[str, tuple[Field, ...]] = {
    "customers": CUSTOMER_FIELDS,
    "orders": ORDER_FIELDS,
    "products": PRODUCT_FIELDS,
}
