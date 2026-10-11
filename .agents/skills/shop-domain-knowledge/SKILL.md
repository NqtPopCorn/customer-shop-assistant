---
name: shop-domain-knowledge
description: "Guidelines and schemas for modeling clothing store catalog data, sizes, stock, and store policies."
---

# Shop Domain Knowledge

Use this skill when reading, expanding, or mocking clothing store catalog data and customer service policies.

## Data Locations
- Product catalog: `data/products.json`
- Store policies: `data/policies.json`
- Data loaders: `src/customer_shop_assistant/loaders/`

## Product Schema (`data/products.json`)
Products should follow this lightweight structure:
```json
{
  "id": "prod-001",
  "name": "Cotton Crewneck T-Shirt",
  "category": "tops",
  "price": 250000,
  "currency": "VND",
  "sizes": ["S", "M", "L", "XL"],
  "colors": ["White", "Black", "Navy"],
  "material": "100% Cotton",
  "in_stock": true,
  "description": "Breathable daily cotton tee for casual wear."
}
```

## Policy Schema (`data/policies.json`)
Store policies must cover frequent customer inquiries:
```json
{
  "shipping": {
    "standard_delivery": "2-3 business days nationwide",
    "fee": 30000,
    "free_shipping_threshold": 500000
  },
  "return_and_exchange": {
    "return_window_days": 7,
    "conditions": "Unworn with tags intact and original receipt"
  },
  "size_guide": {
    "S": "Height 155-165cm, Weight 48-55kg",
    "M": "Height 165-172cm, Weight 55-65kg",
    "L": "Height 170-178cm, Weight 65-75kg",
    "XL": "Height 175-185cm, Weight 75-85kg"
  }
}
```

## Implementation Rules
1. Keep mock data human-readable and focused on clothing store retail scenarios.
2. In `src/customer_shop_assistant/loaders/`, load files using `settings.DATA_DIR` so paths are never hardcoded.
