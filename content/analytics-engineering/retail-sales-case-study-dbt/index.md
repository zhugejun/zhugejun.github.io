---
title: "From Kimball to Code: Implementing the Retail Star Schema in dbt"
date: "2026-01-29"
categories:
  - Analytics Engineering
  - Data Warehousing
  - Dimensional Modeling
tags:
  - dbt
  - kimball
  - star-schema
  - fact-tables
  - dimension-tables
  - data-modeling
---


If you're learning dimensional modeling, Chapter 3 of Ralph Kimball's *The Data Warehouse Toolkit* is where the magic really begins. The retail sales case study is a masterclass in designing fact and dimension tables.

But theory is only half the battle. In this post, I'll show you how to **implement these concepts using dbt (data build tool)**—the modern standard for analytics engineering.

---

## The Business Scenario

The chapter presents a typical retail scenario: a grocery store chain wants to analyze sales data. The questions the business wants to answer are familiar:

- What products are selling?
- Which stores perform best?
- How effective are promotions?
- What are sales trends over time?

---

## The Four-Step Dimensional Design Process

Kimball introduces a systematic approach:

1. **Select the business process** → Retail sales transactions
2. **Declare the grain** → One row per product per transaction (the most atomic level)
3. **Identify the dimensions** → Date, Product, Store, Promotion, etc.
4. **Identify the facts** → Sales quantity, sales dollar amount, cost, profit

**Key Insight:** Getting the grain right is *everything*. If you declare the grain too high (e.g., daily totals by store), you lose the ability to drill down.

---

## Mapping Kimball to dbt: Project Structure

Here's how to organize your dbt project to reflect Kimball's methodology:

```
models/
├── staging/                    # Clean source data
│   ├── stg_pos_transactions.sql
│   ├── stg_products.sql
│   ├── stg_stores.sql
│   └── stg_promotions.sql
├── intermediate/               # Business logic & transformations
│   └── int_sales_enriched.sql
├── marts/
│   ├── core/
│   │   ├── dim_date.sql
│   │   ├── dim_product.sql
│   │   ├── dim_store.sql
│   │   ├── dim_promotion.sql
│   │   └── fct_sales.sql
│   └── marketing/
│       └── fct_promotion_performance.sql
```

**dbt Naming Conventions that Align with Kimball:**

| Kimball Concept | dbt Naming Convention |
|-----------------|----------------------|
| Fact Table | `fct_*` |
| Dimension Table | `dim_*` |
| Staging Layer | `stg_*` |
| Intermediate | `int_*` |

---

## Designing and Building Dimension Tables in dbt

### Date Dimension

The date dimension is unique—it's typically generated rather than sourced from transactional data.

**`models/marts/core/dim_date.sql`**

```sql
{{
    config(
        materialized='table',
        tags=['dimension', 'core']
    )
}}

with date_spine as (
    {{ dbt_utils.date_spine(
        datepart="day",
        start_date="cast('2020-01-01' as date)",
        end_date="cast('2030-12-31' as date)"
    ) }}
),

date_dimension as (
    select
        -- Surrogate key
        {{ dbt_utils.generate_surrogate_key(['date_day']) }} as date_key,
        
        -- Date attributes
        date_day as date_actual,
        extract(year from date_day) as year,
        extract(month from date_day) as month,
        extract(day from date_day) as day_of_month,
        extract(dayofweek from date_day) as day_of_week,
        {{ dbt_date.day_name('date_day') }} as day_name,
        {{ dbt_date.month_name('date_day') }} as month_name,
        
        -- Fiscal calendar (assuming fiscal year starts in February)
        case 
            when extract(month from date_day) >= 2 
            then extract(year from date_day)
            else extract(year from date_day) - 1
        end as fiscal_year,
        
        -- Useful flags
        case 
            when extract(dayofweek from date_day) in (1, 7) then true 
            else false 
        end as is_weekend,
        
        -- Add holiday logic as needed
        false as is_holiday
        
    from date_spine
)

select * from date_dimension
```

**Lesson:** Pre-build your date dimension with all calendar attributes. This avoids complex SQL date functions in downstream queries.

---

### Product Dimension

**`models/marts/core/dim_product.sql`**

```sql
{{
    config(
        materialized='table',
        tags=['dimension', 'core']
    )
}}

with source_products as (
    select * from {{ ref('stg_products') }}
),

product_dimension as (
    select
        -- Surrogate key (insulates from source system changes)
        {{ dbt_utils.generate_surrogate_key(['sku']) }} as product_key,
        
        -- Natural key (kept for reference, not for joining)
        sku,
        
        -- Product attributes
        product_name,
        brand_name,
        category_name,
        department_name,
        
        -- Package attributes
        package_type,
        package_size,
        
        -- Product characteristics (grocery-specific)
        fat_content,
        diet_type,
        shelf_life_days,
        
        -- Flatten the hierarchy (avoid snowflaking!)
        brand_name as brand,
        category_name as subcategory,
        department_name as category,
        
        -- Metadata
        current_timestamp() as dbt_loaded_at
        
    from source_products
)

select * from product_dimension
```

**Key Kimball Principle in dbt:** Notice how we **flatten the hierarchy** into a single dimension table. In traditional normalized databases, you might have separate `brand`, `category`, and `department` tables. In dimensional modeling, we denormalize these into one wide table. This makes queries simpler and faster.

---

### Store Dimension

**`models/marts/core/dim_store.sql`**

```sql
{{
    config(
        materialized='table',
        tags=['dimension', 'core']
    )
}}

with source_stores as (
    select * from {{ ref('stg_stores') }}
),

store_dimension as (
    select
        -- Surrogate key
        {{ dbt_utils.generate_surrogate_key(['store_number']) }} as store_key,
        
        -- Natural key
        store_number,
        
        -- Store attributes
        store_name,
        store_manager,
        
        -- Location hierarchy (flattened)
        street_address,
        city,
        state,
        zip_code,
        country,
        
        -- Geographic hierarchy
        district_name,
        region_name,
        
        -- Store characteristics
        selling_square_footage,
        first_open_date,
        last_remodel_date,
        
        -- Metadata
        current_timestamp() as dbt_loaded_at
        
    from source_stores
)

select * from store_dimension
```

---

### Promotion Dimension

**`models/marts/core/dim_promotion.sql`**

```sql
{{
    config(
        materialized='table',
        tags=['dimension', 'core']
    )
}}

with source_promotions as (
    select * from {{ ref('stg_promotions') }}
),

-- Handle "No Promotion" case (Kimball's null handling principle)
no_promotion_row as (
    select
        {{ dbt_utils.generate_surrogate_key(["'NO_PROMOTION'"]) }} as promotion_key,
        'NO_PROMOTION' as promotion_id,
        'No Promotion' as promotion_name,
        'None' as price_reduction_type,
        'None' as ad_type,
        'None' as display_type,
        'None' as coupon_type,
        cast(null as date) as promotion_start_date,
        cast(null as date) as promotion_end_date,
        current_timestamp() as dbt_loaded_at
),

promotion_dimension as (
    select
        {{ dbt_utils.generate_surrogate_key(['promotion_id']) }} as promotion_key,
        promotion_id,
        promotion_name,
        price_reduction_type,
        ad_type,
        display_type,
        coupon_type,
        promotion_start_date,
        promotion_end_date,
        current_timestamp() as dbt_loaded_at
        
    from source_promotions
)

-- Union actual promotions with "No Promotion" row
select * from no_promotion_row
union all
select * from promotion_dimension
```

**Key Kimball Principle:** Instead of using NULLs in the fact table's foreign keys, we create a "No Promotion" row. This ensures referential integrity and makes queries cleaner.

---

## Designing and Building the Fact Table in dbt

**`models/marts/core/fct_sales.sql`**

```sql
{{
    config(
        materialized='incremental',
        unique_key='sales_line_key',
        tags=['fact', 'core']
    )
}}

with source_transactions as (
    select * from {{ ref('stg_pos_transactions') }}
    {% if is_incremental() %}
    where transaction_date > (select max(transaction_date) from {{ this }})
    {% endif %}
),

-- Join to get dimension keys
sales_with_keys as (
    select
        -- Degenerate dimension (no separate table needed)
        t.transaction_number,
        t.line_number,
        
        -- Foreign keys to dimensions
        d.date_key,
        p.product_key,
        s.store_key,
        coalesce(pr.promotion_key, no_promo.promotion_key) as promotion_key,
        
        -- Facts (measures) - all additive
        t.sales_quantity,
        t.sales_dollar_amount,
        t.cost_dollar_amount,
        t.sales_dollar_amount - t.cost_dollar_amount as gross_profit,
        
        -- Additional context
        t.transaction_date,
        t.transaction_time
        
    from source_transactions t
    
    -- Join to dimensions using natural keys, retrieve surrogate keys
    left join {{ ref('dim_date') }} d
        on t.transaction_date = d.date_actual
    
    left join {{ ref('dim_product') }} p
        on t.sku = p.sku
    
    left join {{ ref('dim_store') }} s
        on t.store_number = s.store_number
    
    left join {{ ref('dim_promotion') }} pr
        on t.promotion_id = pr.promotion_id
    
    -- Get the "No Promotion" key for nulls
    left join {{ ref('dim_promotion') }} no_promo
        on no_promo.promotion_id = 'NO_PROMOTION'
),

final as (
    select
        -- Create a unique key for the fact row
        {{ dbt_utils.generate_surrogate_key([
            'transaction_number', 
            'line_number'
        ]) }} as sales_line_key,
        
        -- Dimension keys
        date_key,
        product_key,
        store_key,
        promotion_key,
        
        -- Degenerate dimension
        transaction_number,
        
        -- Facts
        sales_quantity,
        sales_dollar_amount,
        cost_dollar_amount,
        gross_profit,
        
        -- Metadata
        transaction_date,
        transaction_time,
        current_timestamp() as dbt_loaded_at
        
    from sales_with_keys
)

select * from final
```

### Key Concepts Implemented:

| Kimball Concept | dbt Implementation |
|-----------------|-------------------|
| **Grain** | One row per transaction line item (enforced by `unique_key`) |
| **Surrogate Keys** | `dbt_utils.generate_surrogate_key()` |
| **Degenerate Dimension** | `transaction_number` stored directly in fact table |
| **Additive Facts** | `sales_quantity`, `sales_dollar_amount`, `cost_dollar_amount`, `gross_profit` |
| **Null Handling** | `coalesce()` to "No Promotion" key |
| **Incremental Loading** | `is_incremental()` for efficiency |

---

## Implementing Slowly Changing Dimensions (SCD Type 2) in dbt

When a product's brand changes or a store gets a new manager, we need to track history. dbt makes SCD Type 2 easy with **snapshots**.

**`snapshots/snap_product.sql`**

```sql
{% snapshot snap_product %}

{{
    config(
        target_schema='snapshots',
        unique_key='sku',
        strategy='check',
        check_cols=['brand_name', 'category_name', 'department_name', 'product_name']
    )
}}

select * from {{ source('raw', 'products') }}

{% endsnapshot %}
```

Then reference the snapshot in your dimension:

**`models/marts/core/dim_product_scd2.sql`**

```sql
{{
    config(
        materialized='table'
    )
}}

with snapshot_data as (
    select * from {{ ref('snap_product') }}
),

final as (
    select
        {{ dbt_utils.generate_surrogate_key(['sku', 'dbt_valid_from']) }} as product_key,
        sku,
        product_name,
        brand_name,
        category_name,
        department_name,
        
        -- SCD Type 2 tracking columns
        dbt_valid_from as valid_from,
        dbt_valid_to as valid_to,
        case when dbt_valid_to is null then true else false end as is_current
        
    from snapshot_data
)

select * from final
```

---

## Configuring Your `dbt_project.yml`

```yaml
name: 'retail_dwh'
version: '1.0.0'

config-version: 2

model-paths: ["models"]
snapshot-paths: ["snapshots"]
test-paths: ["tests"]

vars:
  # Default "unknown" values for dimensions
  unknown_product_key: '0000000000000000'
  unknown_store_key: '0000000000000000'

models:
  retail_dwh:
    staging:
      +materialized: view
      +schema: staging
    intermediate:
      +materialized: ephemeral
    marts:
      core:
        +materialized: table
        +schema: marts
      marketing:
        +materialized: table
        +schema: marts_marketing
```

---

## Adding Data Quality Tests

Kimball emphasizes data quality. dbt's testing framework makes this easy.

**`models/marts/core/schema.yml`**

```yaml
version: 2

models:
  - name: fct_sales
    description: "Retail sales fact table at the transaction line item grain"
    columns:
      - name: sales_line_key
        description: "Surrogate key for the fact row"
        tests:
          - unique
          - not_null
      
      - name: date_key
        description: "Foreign key to dim_date"
        tests:
          - not_null
          - relationships:
              to: ref('dim_date')
              field: date_key
      
      - name: product_key
        description: "Foreign key to dim_product"
        tests:
          - not_null
          - relationships:
              to: ref('dim_product')
              field: product_key
      
      - name: store_key
        description: "Foreign key to dim_store"
        tests:
          - not_null
          - relationships:
              to: ref('dim_store')
              field: store_key
      
      - name: promotion_key
        description: "Foreign key to dim_promotion"
        tests:
          - not_null
          - relationships:
              to: ref('dim_promotion')
              field: promotion_key
      
      - name: sales_dollar_amount
        description: "Total sales amount in dollars"
        tests:
          - not_null
          - dbt_utils.accepted_range:
              min_value: 0

  - name: dim_date
    description: "Conformed date dimension"
    columns:
      - name: date_key
        tests:
          - unique
          - not_null

  - name: dim_product
    description: "Product dimension with flattened hierarchy"
    columns:
      - name: product_key
        tests:
          - unique
          - not_null
      - name: sku
        tests:
          - unique
          - not_null
```

---

## The Star Schema in dbt: DAG Visualization

When you run `dbt docs generate` and view the lineage graph, you'll see the star schema emerge:

```
stg_pos_transactions ──┐
                       │
stg_products ──► dim_product ──┐
                               │
stg_stores ──► dim_store ──────┼──► fct_sales
                               │
stg_promotions ──► dim_promotion ──┘
                               │
              dim_date ────────┘
```

---

## Querying the Star Schema

Once built, your analysts can write intuitive queries:

```sql
-- Sales by product category and month
select
    d.year,
    d.month_name,
    p.department_name,
    p.category_name,
    sum(f.sales_dollar_amount) as total_sales,
    sum(f.gross_profit) as total_profit
from {{ ref('fct_sales') }} f
join {{ ref('dim_date') }} d on f.date_key = d.date_key
join {{ ref('dim_product') }} p on f.product_key = p.product_key
group by 1, 2, 3, 4
order by 1, 2, total_sales desc
```

```sql
-- Promotion effectiveness analysis
select
    pr.promotion_name,
    pr.ad_type,
    pr.display_type,
    count(distinct f.transaction_number) as transaction_count,
    sum(f.sales_quantity) as units_sold,
    sum(f.sales_dollar_amount) as total_revenue,
    sum(f.gross_profit) as total_profit,
    sum(f.gross_profit) / nullif(sum(f.sales_dollar_amount), 0) as profit_margin
from {{ ref('fct_sales') }} f
join {{ ref('dim_promotion') }} pr on f.promotion_key = pr.promotion_key
where pr.promotion_id != 'NO_PROMOTION'
group by 1, 2, 3
order by total_revenue desc
```

---

## Key Takeaways: Kimball + dbt

| Kimball Principle | dbt Implementation |
|-------------------|-------------------|
| **Declare the grain** | Enforce with `unique_key` in config and uniqueness tests |
| **Use surrogate keys** | `dbt_utils.generate_surrogate_key()` |
| **Avoid nulls in FKs** | Create "Unknown" or "N/A" dimension rows |
| **Denormalize dimensions** | Flatten hierarchies in dimension models |
| **Track history (SCD)** | dbt snapshots for Type 2 |
| **Conformed dimensions** | Share `dim_date`, `dim_product` across marts via `ref()` |
| **Data quality** | dbt tests for uniqueness, not null, relationships |
| **Documentation** | `schema.yml` descriptions and `dbt docs` |

---

## Final Thoughts

The retail case study in Chapter 3 of *The Data Warehouse Toolkit* isn't just theory—it's a blueprint you can implement today using dbt. The combination of Kimball's time-tested dimensional modeling principles with dbt's modern tooling gives you:

- **Version-controlled data transformations**
- **Automated testing and documentation**
- **Reproducible builds**
- **Clear lineage and dependencies**

Master these patterns, and you'll have the foundation for building effective, maintainable data warehouses that analysts actually want to use.

---

## Resources

- [The Data Warehouse Toolkit](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/books/data-warehouse-dbt-toolkit/) by Ralph Kimball
- [dbt Documentation](https://docs.getdbt.com/)
- [dbt-utils Package](https://hub.getdbt.com/dbt-labs/dbt_utils/latest/)
- [dbt Snapshots (SCD Type 2)](https://docs.getdbt.com/docs/build/snapshots)

---

*Happy modeling!*