---
title: "Fundamentals of dbt"
date: "2026-01-21"
categories:
  - Analytics Engineering
  - dbt
tags:
  - dbt
  - analytics
  - pipeline
---

This cheat sheet covers the fundamentals of **dbt (data build tool)**—a popular data transformation tool used in modern data engineering and analytics workflows. It includes key concepts, commands, and best practices to get you started with dbt.

## Fundamentals (Must Know)

### What is dbt?

**dbt (data build tool)** is a transformation tool that enables analysts and engineers to transform data in their warehouse using SQL.

**Key concept:** dbt handles the **T** in ELT (Extract, Load, Transform). It doesn't extract or load data—it transforms data that's already in your warehouse.

**What dbt does:**

- Turns SELECT statements into tables/views
- Manages dependencies between models
- Tests data quality
- Generates documentation
- Enables version control for transformations

**What dbt doesn't do:**

- Extract data from sources
- Load data into the warehouse
- Orchestrate pipelines (though dbt Cloud has scheduling)

---

### How does dbt work?

1. You write **SELECT statements** (called models) in `.sql` files
2. dbt compiles these into full SQL (resolving references, applying Jinja)
3. dbt determines the **correct execution order** based on dependencies
4. dbt runs the SQL against your data warehouse
5. Tables/views are created in the warehouse

```sql
-- models/customers.sql
SELECT
    id,
    first_name,
    last_name,
    created_at
FROM {{ ref('stg_customers') }}
WHERE is_active = true
```

dbt compiles this to:

```sql
CREATE VIEW analytics.customers AS (
    SELECT
        id,
        first_name,
        last_name,
        created_at
    FROM analytics.stg_customers
    WHERE is_active = true
)
```

---

### What are models?

Models are **SQL SELECT statements** that define a transformation. Each model lives in a `.sql` file and creates one table or view in your warehouse.

**Model file:** `models/dim_customers.sql`

```sql
SELECT
    customer_id,
    customer_name,
    segment
FROM {{ ref('stg_customers') }}
```

**Models are typically organized in layers:**

|Layer|Prefix|Purpose|
|---|---|---|
|**Staging**|`stg_`|Light transformations, 1:1 with source tables|
|**Intermediate**|`int_`|Business logic, joining staging models|
|**Marts**|`dim_`, `fct_`|Final tables for BI tools and analysts|

---

## What is the ref() function?

`ref()` is dbt's most important function. It creates **dependencies between models** and handles schema/database references automatically.

```sql
-- This creates a dependency on stg_orders
SELECT * FROM {{ ref('stg_orders') }}
```

**Why use ref():**

- dbt builds a DAG (dependency graph) from refs
- Ensures models run in correct order
- Handles environment-specific schemas (dev vs prod)
- Enables impact analysis

**Never hardcode table names.** Always use `ref()` for models and `source()` for raw tables.

---

## What are sources?

Sources define your **raw data tables** that exist in the warehouse before dbt runs. They're declared in YAML files.

```yaml
# models/staging/sources.yml
version: 2

sources:
  - name: stripe
    database: raw
    schema: stripe_data
    tables:
      - name: payments
        description: Raw payment data from Stripe
        loaded_at_field: _loaded_at
        freshness:
          warn_after: {count: 12, period: hour}
          error_after: {count: 24, period: hour}
```

The `freshness` key lets dbt check if your source data is stale. It compares the current time against the most recent value in `loaded_at_field`.

**Reference in SQL:**

```sql
SELECT * FROM {{ source('stripe', 'payments') }}
```

The source `name` is essentially an **alias** that represents the `database.schema` combination.

**Benefits:**

- Documents raw data
- Freshness checks
- Single place to update if source changes
- Clear lineage from raw → transformed

**Other common keys:**

```yaml
sources:
  - name: stripe
    database: raw
    schema: stripe_data
    description: "Data from Stripe API"
    
    # Default freshness for all tables in this source
    freshness:
      warn_after: {count: 12, period: hour}
    loaded_at_field: _loaded_at
    
    # Quoting options (useful for case-sensitive warehouses)
    quoting:
      database: true
      schema: true
      identifier: true

    tables:
      - name: payments
        description: "Raw payment data"
        identifier: actual_table_name  # if physical name differs
        
        # Override source-level freshness
        freshness:
          error_after: {count: 24, period: hour}
        loaded_at_field: _etl_loaded_at
        
        # Column definitions
        columns:
          - name: payment_id
            description: "Primary key"
            data_tests:
              - unique
              - not_null
          - name: amount
            description: "Payment amount in cents"
        
        # Table-level tests
        data_tests: 
          - dbt_utils.at_least_one: 
              column_name: payment_id

      - name: customers
        freshness: null  # disable freshness check for this table
```

| Level  | Key               | Purpose                                                                 |
| ------ | ----------------- | ----------------------------------------------------------------------- |
| Source | `name`            | Logical alias                                                           |
| Source | `database`        | Physical database                                                       |
| Source | `schema`          | Physical schema                                                         |
| Source | `description`     | Documentation                                                           |
| Source | `freshness`       | Default freshness for all tables                                        |
| Source | `loaded_at_field` | Default timestamp column                                                |
| Source | `quoting`         | Control identifier quoting                                              |
| Table  | `name`            | Table name (used in `source()`)                                         |
| Table  | `identifier`      | Actual physical table name if different                                 |
| Table  | `description`     | Documentation                                                           |
| Table  | `freshness`       | Override source-level freshness (compare with `loaded_at_field` column) |
| Table  | `columns`         | Column definitions and tests                                            |
| Table  | `data_tests`      | Table-level tests                                                       |

---

### What are the materializations? Difference between view and table?

Materialization determines **how dbt builds your model** in the warehouse.

|Materialization|What it creates|When to use|
|---|---|---|
|**view**|SQL view|Small transformations, always need fresh data|
|**table**|Physical table|Larger datasets, frequently queried|
|**incremental**|Table with appended rows|Very large tables, event data|
|**ephemeral**|CTE (no object created)|Reusable logic, don't need to query directly|

**View vs Table:**

|Aspect|View|Table|
|---|---|---|
|Storage|None (just SQL definition)|Stores all data|
|Query speed|Slower (runs transformation each time)|Faster (data pre-computed)|
|Build time|Instant|Takes time to load data|
|Freshness|Always current|Current as of last dbt run|

```sql
-- Set in model file
{{ config(materialized='table') }}

SELECT * FROM {{ ref('stg_orders') }}
```

Or in `dbt_project.yml`:

```yaml
models:
  my_project:
    staging:
      +materialized: view
    marts:
      +materialized: table
```

---

### What is the dbt project structure?

```txt
my_dbt_project/
├── dbt_project.yml        # Project configuration
├── profiles.yml           # Connection credentials (usually in ~/.dbt/)
├── models/
│   ├── staging/           # Staging models
│   │   ├── stg_customers.sql
│   │   ├── stg_orders.sql
│   │   └── sources.yml
│   ├── intermediate/      # Intermediate models
│   │   └── int_orders_grouped.sql
│   └── marts/             # Final models
│       ├── dim_customers.sql
│       ├── fct_orders.sql
│       └── schema.yml     # Tests and docs
├── tests/                 # Custom tests
├── macros/                # Reusable SQL/Jinja
├── seeds/                 # CSV files to load
├── snapshots/             # SCD Type 2 snapshots
└── analyses/              # Ad-hoc queries (not materialized)
```

---

### What are dbt tests?

Tests are **assertions about your data**. dbt runs them after building models and alerts you to failures.

**Built-in generic tests:**

```yaml
# models/schema.yml
version: 2

models:
  - name: dim_customers
    columns:
      - name: customer_id
        tests:
          - unique
          - not_null
      - name: status
        tests:
          - accepted_values:
              values: ['active', 'inactive', 'pending']
      - name: country_id
        tests:
          - relationships:
              to: ref('dim_countries')
              field: country_id
```

1. `unique` - Ensures `customer_id` has no duplicates.
2. `not_null` - Ensures `customer_id` is never null.
3. `accepted_values` - Ensures `status` only contains specified values.
4. `relationships` - It verifies that every `country_id` in your source table exists in `dim_countries.country_id`.

**Custom tests (singular tests):**

```sql
-- tests/assert_positive_revenue.sql
SELECT order_id, revenue
FROM {{ ref('fct_orders') }}
WHERE revenue < 0
-- Test passes if this returns 0 rows
```

**Custom test with macros:**

```sql
-- macros/test_is_positive.sql

{% test is_positive(model, column_name) %}

SELECT *
FROM {{ model }}
WHERE {{ column_name }} < 0

{% endtest %}
```

Then use it like a built-in test:

```yaml
columns:
  - name: amount
    tests:
      - is_positive
```

**Test severity:**

```yaml
- name: customer_id
  tests:
    - unique:
        severity: error    # Fails the run
    - not_null:
        severity: warn     # Warning only
```

---

### What is documentation in dbt?

dbt generates **documentation websites** from your YAML files and model descriptions.

```yaml
# models/schema.yml
version: 2

models:
  - name: fct_orders
    description: >
      Order fact table. One row per order.
      Grain: order_id
    columns:
      - name: order_id
        description: Primary key
      - name: customer_id
        description: Foreign key to dim_customers
      - name: order_total
        description: Total order value in USD
```

**Doc blocks for longer descriptions:**

```markdown
-- models/docs.md
{% docs order_status %}
Order status can be one of:
- **pending**: Order placed but not paid
- **paid**: Payment received
- **shipped**: Order dispatched
- **delivered**: Order received by customer
{% enddocs %}
```

```yaml
- name: status
  description: '{{ doc("order_status") }}'
```

**Generate and serve docs:**

```bash
dbt docs generate
dbt docs serve
```

---

## Intermediate (Good to Know)

### What are incremental models?

Incremental models only transform **new or changed data** instead of rebuilding the entire table. Essential for large tables.

```sql
{{ config(
    materialized='incremental',
    unique_key='event_id'
) }}

SELECT
    event_id,
    user_id,
    event_type,
    event_timestamp
FROM {{ source('events', 'raw_events') }}

{% if is_incremental() %}
    WHERE event_timestamp > (SELECT MAX(event_timestamp) FROM {{ this }})
{% endif %}
```

**Key concepts:**

- `is_incremental()`: True when running incrementally (not full refresh)
- `{{ this }}`: Refers to the current model's existing table
- `unique_key`: Handles updates (upserts) to existing rows

**When to use:**

- Event/log data (millions of rows)
- Tables that take too long to rebuild fully
- Data with clear timestamp or ID for filtering

**Force full refresh:**

```bash
dbt run --full-refresh -s my_incremental_model
```

---

### What are snapshots?

Snapshots implement **Slowly Changing Dimension Type 2 (SCD2)**—they track historical changes to records over time.

```sql
-- snapshots/customer_snapshot.sql
{% snapshot customers_snapshot %}

{{
    config(
        target_schema='snapshots',
        unique_key='customer_id',
        strategy='timestamp',
        updated_at='updated_at'
    )
}}

SELECT * FROM {{ source('app', 'customers') }}

{% endsnapshot %}
```

**Result:** A table with `dbt_valid_from`, `dbt_valid_to`, and `dbt_scd_id` columns tracking each version of a record.

**Strategies:**

- `timestamp`: Uses a timestamp column to detect changes
- `check`: Compares specific columns for changes

```bash
dbt snapshot
```

---

### What are macros?

Macros are **reusable pieces of SQL/Jinja** code. Like functions for your SQL.

```sql
-- macros/cents_to_dollars.sql
{% macro cents_to_dollars(column_name) %}
    ({{ column_name }} / 100)::decimal(10,2)
{% endmacro %}
```

**Use in models:**

```sql
SELECT
    order_id,
    {{ cents_to_dollars('amount_cents') }} AS amount_dollars
FROM {{ ref('stg_orders') }}
```

**Common use cases:**

- Unit conversions
- Date formatting
- Environment-specific logic
- DRY (Don't Repeat Yourself) SQL patterns

---

### What is Jinja?

Jinja is a **templating language** that dbt uses to make SQL dynamic.

**Variables:**

```sql
{% set payment_methods = ['credit_card', 'paypal', 'bank_transfer'] %}

SELECT
    order_id,
    {% for method in payment_methods %}
        SUM(CASE WHEN payment_method = '{{ method }}' THEN amount ELSE 0 END) AS {{ method }}_amount
        {% if not loop.last %},{% endif %}
    {% endfor %}
FROM {{ ref('stg_payments') }}
GROUP BY 1
```

**Common Jinja features:**

- `{{ }}` - Print expressions
- `{% %}` - Logic (if/for/set)
- `{# #}` - Comments
- `ref()`, `source()`, `config()` - dbt functions

**Conditionals:**

```sql
SELECT
    *,
    {% if target.name == 'prod' %}
        sensitive_column
    {% else %}
        'REDACTED' AS sensitive_column
    {% endif %}
FROM {{ ref('stg_users') }}
```

---

### What are packages?

Packages are **reusable dbt projects** you can install and use. Like libraries in Python.

**Install packages in `packages.yml`:**

```yaml
packages:
  - package: dbt-labs/dbt_utils
    version: 1.1.1
  - package: dbt-labs/codegen
    version: 0.12.1
  - git: https://github.com/company/internal-package.git
    revision: v1.0.0
```

```bash
dbt deps  # Install packages
```

**Popular packages:**

|Package|Purpose|
|---|---|
|`dbt_utils`|Generic tests, macros (surrogate keys, pivots, etc.)|
|`codegen`|Generate YAML and SQL automatically|
|`dbt_expectations`|Great Expectations-style tests|
|`audit_helper`|Compare tables during refactoring|
|`dbt_date`|Date dimension and utilities|

**Using dbt_utils:**

```sql
SELECT
    {{ dbt_utils.generate_surrogate_key(['customer_id', 'order_date']) }} AS order_key,
    *
FROM {{ ref('stg_orders') }}
```

---

### What are seeds?

Seeds are **CSV files** that dbt loads into your warehouse as tables. Good for small, static reference data.

```txt
seeds/
└── country_codes.csv
```

```csv
country_code,country_name,region
US,United States,North America
CA,Canada,North America
UK,United Kingdom,Europe
```

```bash
dbt seed
```

**Use cases:**

- Country/state mappings
- Category lookups
- Test data
- Static configuration

**Reference like any model:**

```sql
SELECT * FROM {{ ref('country_codes') }}
```

**Not for:** Large datasets, frequently changing data, sensitive data.

---

### What are hooks?

Hooks run **SQL before or after** dbt operations.

**Types:**

- `pre-hook`: Runs before a model is built
- `post-hook`: Runs after a model is built
- `on-run-start`: Runs at start of `dbt run`
- `on-run-end`: Runs at end of `dbt run`

```sql
-- In model config
{{ config(
    post_hook="GRANT SELECT ON {{ this }} TO ROLE analyst"
) }}

SELECT * FROM {{ ref('stg_orders') }}
```

**In dbt_project.yml:**

```yaml
on-run-start:
  - "CREATE SCHEMA IF NOT EXISTS {{ target.schema }}"

on-run-end:
  - "GRANT USAGE ON SCHEMA {{ target.schema }} TO ROLE analyst"

models:
  my_project:
    marts:
      +post-hook:
        - "ANALYZE TABLE {{ this }}"
```

---

### What is the dbt DAG?

The DAG (Directed Acyclic Graph) represents **dependencies between all models**.

```txt
sources → staging → intermediate → marts
                                      ↓
                                  exposures
```

dbt builds the DAG by parsing `ref()` and `source()` functions.

**Why it matters:**

- Determines execution order
- Enables selective runs (`dbt run -s model+` runs model and all downstream)
- Powers lineage in documentation
- Impact analysis (what breaks if I change this?)

**Selection syntax:**

```bash
dbt run -s my_model          # Just this model
dbt run -s my_model+         # Model and all downstream
dbt run -s +my_model         # Model and all upstream
dbt run -s +my_model+        # Model, upstream, and downstream
dbt run -s tag:finance       # All models with tag
dbt run -s staging.*         # All models in staging folder
```

---

## Advanced (Senior/Specialized Roles)

### dbt Cloud vs dbt Core

|Feature|dbt Core|dbt Cloud|
|---|---|---|
|**Price**|Free (open source)|Paid (free tier available)|
|**Execution**|CLI on your machine/server|Managed in cloud|
|**Scheduling**|Need external tool (Airflow, cron)|Built-in scheduler|
|**IDE**|Your own (VS Code, etc.)|Browser-based IDE|
|**CI/CD**|Set up yourself|Built-in|
|**Documentation**|Self-hosted|Hosted for you|
|**Logging**|DIY|Built-in history and logs|

**When to use Core:** Small teams, already have orchestration, cost-sensitive.

**When to use Cloud:** Faster setup, need scheduling, want managed infrastructure.

---

### How do you handle CI/CD with dbt?

**Slim CI (dbt Cloud or custom):** Only test models that changed + their downstream dependencies.

```bash
# Get modified models compared to main branch
dbt run -s state:modified+ --state ./target
dbt test -s state:modified+ --state ./target
```

**GitHub Actions example:**

```yaml
name: dbt CI

on:
  pull_request:
    branches: [main]

jobs:
  dbt-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dbt
        run: pip install dbt-snowflake
        
      - name: Run dbt
        run: |
          dbt deps
          dbt build --select state:modified+ --state ./prod-artifacts
        env:
          DBT_PROFILES_DIR: ./
```

**Best practices:**

- Run tests on every PR
- Use separate dev/staging schemas per PR
- Compare results with production using `audit_helper`

---

### What are exposures?

Exposures define **downstream uses of your dbt models**—dashboards, ML models, applications.

```yaml
# models/exposures.yml
version: 2

exposures:
  - name: weekly_revenue_dashboard
    type: dashboard
    maturity: high
    owner:
      name: Data Team
      email: data@company.com
    depends_on:
      - ref('fct_orders')
      - ref('dim_customers')
    url: https://looker.company.com/dashboards/123
    description: Executive dashboard showing weekly revenue metrics
```

**Why use exposures:**

- Document what depends on your models
- Show in DAG/lineage
- Impact analysis (this dashboard uses that model)

---

### What are metrics (semantic layer)?

dbt metrics define **business metrics as code**—single source of truth for calculations.

```yaml
# models/metrics.yml
version: 2

metrics:
  - name: revenue
    label: Total Revenue
    model: ref('fct_orders')
    description: Sum of order totals
    calculation_method: sum
    expression: order_total
    timestamp: order_date
    time_grains: [day, week, month]
    dimensions:
      - customer_segment
      - region
    filters:
      - field: status
        operator: '='
        value: "'completed'"
```

**Note:** The metrics layer has evolved. dbt now recommends **MetricFlow** (acquired by dbt Labs) for the semantic layer. Check current docs for latest approach.

---

### How do you optimize dbt projects?

**Model performance:**

- Use incremental models for large event tables
- Materialize expensive transformations as tables
- Add clustering keys in Snowflake config
- Avoid `SELECT *` in final models

**Project organization:**

- Follow staging → intermediate → marts pattern
- Use consistent naming conventions
- Keep models focused (single responsibility)
- Document everything

**Run performance:**

```bash
# Run in parallel (default is 1)
dbt run --threads 8

# Only run what's needed
dbt run -s state:modified+
```

**dbt_project.yml optimization:**

```yaml
models:
  my_project:
    staging:
      +materialized: view
    intermediate:
      +materialized: ephemeral  # No tables, just CTEs
    marts:
      +materialized: table

# Snowflake-specific
models:
  my_project:
    marts:
      +materialized: table
      +snowflake_warehouse: transforming_wh  # Dedicated warehouse
```

---

### Common debugging commands

```bash
# Compile SQL without running (see what dbt generates)
dbt compile -s my_model

# Show compiled SQL
cat target/compiled/my_project/models/my_model.sql

# Run with debug logging
dbt --debug run -s my_model

# Test a single model
dbt test -s my_model

# List all models that would run
dbt ls -s my_model+

# Check project for errors
dbt parse
```

---

## Quick Reference

## Common Commands

```bash
# Core commands
dbt run                    # Run all models
dbt test                   # Run all tests
dbt build                  # Run + test in DAG order
dbt compile                # Compile without executing

# Selection
dbt run -s my_model        # Single model
dbt run -s +my_model       # Model + upstream
dbt run -s my_model+       # Model + downstream
dbt run -s tag:finance     # By tag
dbt run -s path:models/staging  # By path
dbt run --exclude my_model # Exclude model

# Other
dbt deps                   # Install packages
dbt seed                   # Load CSV seeds
dbt snapshot               # Run snapshots
dbt docs generate          # Generate docs
dbt docs serve             # Serve docs locally
dbt clean                  # Delete target/ folder
dbt debug                  # Test connection
```

## Jinja Cheat Sheet

```sql
-- Variables
{% set my_list = ['a', 'b', 'c'] %}
{% set my_var = 'value' %}

-- Loops
{% for item in my_list %}
    {{ item }}{% if not loop.last %},{% endif %}
{% endfor %}

-- Conditionals
{% if target.name == 'prod' %}
    -- production code
{% else %}
    -- dev code
{% endif %}

-- dbt functions
{{ ref('model_name') }}
{{ source('source_name', 'table_name') }}
{{ this }}                            -- Current model
{{ target.name }}                     -- Target name (dev/prod)
{{ target.schema }}                   -- Target schema
{{ var('my_var', 'default') }}        -- Project variable
{{ env_var('MY_ENV_VAR') }}           -- Environment variable
```

## Project Config (dbt_project.yml)

```yaml
name: 'my_project'
version: '1.0.0'
config-version: 2

profile: 'my_project'

model-paths: ["models"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets: ["target", "dbt_packages"]

vars:
  my_var: 'value'

models:
  my_project:
    +materialized: view
    staging:
      +schema: staging
    marts:
      +materialized: table
      +schema: marts
```

## Testing Checklist

- [ ] Primary keys are unique and not null
- [ ] Foreign keys have valid relationships
- [ ] Enum columns use accepted_values
- [ ] Critical columns are not null
- [ ] Row counts are reasonable (custom test)
- [ ] Numeric columns are within expected ranges
- [ ] Timestamps are not in the future

## Naming Conventions

|Object|Convention|Example|
|---|---|---|
|Staging models|`stg_<source>__<table>`|`stg_stripe__payments`|
|Intermediate|`int_<description>`|`int_orders_pivoted`|
|Dimensions|`dim_<entity>`|`dim_customers`|
|Facts|`fct_<event/process>`|`fct_orders`|
|Metrics|`<verb>_<object>`|`count_orders`|
