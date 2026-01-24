---
title: "Kimball vs Inmon: Why Kimball is Better for Analytics"
date: "2026-01-24"
categories:
  - Analytics Engineering
  - Data Warehousing
  - Kimball
  - Inmon
tags:
  - data-warehousing
  - kimball
  - inmon
  - analytics
---

I have been learning dbt recently and wanted to get a better understanding of data warehousing methodologies.
Two of the most well-known ones are the Kimball and Inmon approaches.
In this post, I will explain the key differences between the two and why Kimball is generally considered better for analytics.

## Kimball (Dimensional Modeling)

Kimball's approach uses a **star schema** with two main types of tables:

1. **Fact Tables**: Store measurable events (sales amount, quantity sold, order totals). These are the "what happened" of your business.
2. **Dimension Tables**: Store descriptive context (customer info, product details, dates). These answer "who, what, where, when."

The star schema looks like this:

```txt
              dim_customers
                    |
                    |
dim_products -- fact_sales -- dim_time
                    |
                    |
              dim_locations
```

<!-- ### Example tables -->

**Example tables:**

| fact_sales |              |             |          |              |
| ---------- | ------------ | ----------- | -------- | ------------ |
| date_key   | customer_key | product_key | quantity | sales_amount |
| 20240115   | 101          | 42          | 2        | 59.98        |

| dim_customers |       |         |
| ------------- | ----- | ------- |
| customer_key  | name  | city    |
| 101           | Alice | Houston |

| dim_products |              |          |       |
| ------------ | ------------ | -------- | ----- |
| product_key  | product_name | category | price |
| 42           | Widget A     | Gadgets  | 29.99 |

| dim_date |            |         |
| -------- | ---------- | ------- |
| date_key | date       | month   |
| 20240115 | 2024-01-15 | January |

**Sample query** - "Total sales by city by category":

```sql
SELECT dc.city, dp.category, SUM(fs.sales_amount) AS total_sales
FROM fact_sales fs
JOIN dim_customers dc ON fs.customer_key = dc.customer_key
JOIN dim_products dp ON fs.product_key = dp.product_key
GROUP BY dc.city, dp.category;
```

4 tables, 2 joins, reads like English, and business-user friendly!

## Inmon (3rd Normal Form)

Inmon's approach uses a **3rd Normal Form (3NF)** structure, focusing on data integrity and reducing redundancy. The data is organized into many related tables, which can make querying more complex.

The 3NF schema looks like this:

```txt
customers                categories
    |                        |
    v                        v
 orders                  products
    |                        |
    +-----> order_lines <----+
```

**Example tables:**

| customers   |       |         |
| ----------- | ----- | ------- |
| customer_id | name  | city    |
| 101         | Alice | Houston |

| products   |              |             |
| ---------- | ------------ | ----------- |
| product_id | product_name | category_id |
| 42         | Widget A     | 1           |

| categories  |               |
| ----------- | ------------- |
| category_id | category_name |
| 1           | Gadgets       |

| orders   |             |            |
| -------- | ----------- | ---------- |
| order_id | customer_id | order_date |
| 1001     | 101         | 2024-01-15 |

| order_lines   |          |            |          |            |
| ------------- | -------- | ---------- | -------- | ---------- |
| order_line_id | order_id | product_id | quantity | unit_price |
| 1             | 1001     | 42         | 2        | 29.99      |

**Sample query** - "Total sales by city by category":

```sql
SELECT c.city, cat.category_name, SUM(ol.quantity * ol.unit_price) AS total_sales
FROM order_lines ol
JOIN orders o ON ol.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON ol.product_id = p.product_id
JOIN categories cat ON p.category_id = cat.category_id
GROUP BY c.city, cat.category_name;
```

5 tables, 4 joins, more complex and harder for business users to understand.

## Key Differences

| Aspect        | 3NF               | Dimensional      |
| ------------- | ----------------- | ---------------- |
| Goal          | Data integrity    | Query simplicity |
| Redundancy    | Minimal           | Intentional      |
| Joins         | Many              | Few              |
| Primary Users | Engineers         | Analysts         |
| Typical Layer | Central warehouse | Data marts       |

The fundamental distinction: In 3NF, relationships describe how entities depend on each other.
In a star schema, relationships exist only to describe facts.

## Why Kimball is Better for Analytics

- **Simplicity**: Kimball's star schema is easier to understand and query, making it more accessible for analysts and business users.
- **Performance**: Fewer joins in Kimball's model often lead to faster query performance.
- **Business Focus**: Kimball's approach is designed with business questions in mind, making it more aligned with analytics needs.
- **Flexibility**: Dimensional models can be more easily adapted to changing business requirements.

## References

- [The Data Warehouse Toolkit by Ralph Kimball](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/data-warehouse-dw-toolkit/)
