# Analytics Engineer Blog Post Ideas
### A 6-Month Content Plan for Career Pivoters from Higher Ed

---

## Month 1 — Establish Your Story

### 1. Why I'm Leaving Higher Ed After 10 Years to Become an Analytics Engineer
- What made me realize my analyst role had a ceiling and what "analytics engineer" even means
- The moment I discovered dbt and how it reframed what I thought data work could look like

### 2. What a Higher Ed Analyst Actually Does (And Why It's Better Training Than You Think)
- The real scope of higher ed analytics: enrollment, retention, financial aid, accreditation reporting
- Why working with messy, politically sensitive, compliance-heavy data is excellent preparation for AE work

### 3. The Skills Gap I Discovered When I Started Studying for the Analytics Engineer Role
- Honest breakdown of what I already had (SQL, domain knowledge, stakeholder communication) vs. what I needed (dbt, version control, data modeling theory)
- How I built a 6-month learning plan and what resources I chose and why

---

## Month 1–2 — Show Your SQL Depth

### 4. Window Functions Explained With a Real Student Enrollment Dataset
- What window functions are and why they're different from GROUP BY with a concrete enrollment example
- Walk through ROW_NUMBER, RANK, LAG/LEAD using semester-by-semester student records

### 5. How I Rewrote a 200-Line Query Into Something Readable Using CTEs
- Before and after: what the original query looked like and why it was hard to maintain
- The mental model I use now for breaking a complex query into named, readable steps

### 6. The SQL Habits I Had to Unlearn Coming From Excel-Heavy Environments
- Why writing SQL to match what Excel would do is often the wrong approach (e.g., pivoting too early, filtering too late)
- Practical habits that improved my query performance and readability overnight

### 7. Slowly Changing Dimensions: What They Are and Why Higher Ed Data Is Full of Them
- What a slowly changing dimension is and why student records (major changes, enrollment status, advisor assignments) are a textbook example
- How SCD Type 1 vs Type 2 affects the questions you can and cannot answer downstream

---

## Month 2–4 — Document Your dbt Journey

### 8. dbt for Beginners: My First Model, My First Mistake
- How to set up a dbt project locally and run your first model — exactly what I did step by step
- The mistake I made immediately (and what the error message was trying to tell me)

### 9. How I Structured a dbt Project Around a Fake University Data Warehouse
- How I chose a dataset (IPEDS) and designed a project around it to practice realistic scenarios
- The folder structure I landed on and the reasoning behind staging, intermediate, and mart layers

### 10. Testing in dbt — Why I Wish I Had This When I Was Writing Reports in Excel
- What dbt's built-in tests (not_null, unique, accepted_values, relationships) actually do and how to write them
- A real example where a test caught a data quality issue I would have missed before

### 11. What the dbt Staging → Intermediate → Mart Pattern Actually Means in Plain English
- Why you don't just dump all your logic into one model and what each layer is responsible for
- A concrete walkthrough using student enrollment data from raw source to final reporting mart

### 12. Building a Student Retention Model in dbt From Scratch
- How I defined "retention" (a harder question than it sounds) and modeled it as a dbt mart
- The grain of the model, what columns it has, and how a downstream analyst would use it

### 13. dbt Docs: The Feature That Would Have Saved My Team Hundreds of Hours
- How dbt auto-generates documentation and a data lineage graph from your project
- Why documentation-as-code solves the problem of wikis that are always out of date

---

## Month 3–4 — Data Modeling Fundamentals

### 14. Kimball vs. One Big Table: What I Learned Building My First Star Schema
- The core idea behind dimensional modeling: facts, dimensions, and why the star schema makes querying intuitive
- When a wide, denormalized table is actually fine and when it creates problems at scale

### 15. Grain: The Most Important Word in Data Modeling Nobody Explained to Me
- What "grain" means, why you have to define it before you write a single line of SQL
- A real example where mixing grain in a model caused double-counting that took days to find

### 16. How I Modeled Enrollment History to Answer "How Many Students Were Enrolled on Any Given Day?"
- Why this question is deceptively hard and how a spine/calendar approach solves it
- The SQL and dbt model pattern I used and how it can be adapted to other point-in-time questions

### 17. Naming Conventions in a Data Warehouse — Why It Matters More Than You Think
- The naming patterns I adopted (stg_, int_, fct_, dim_) and what each signals about a model's purpose
- How consistent naming makes onboarding new team members dramatically easier

---

## Month 4–5 — The Broader AE Toolkit

### 18. Git for Analytics Engineers Who Are Not Developers (A Practical Guide)
- The five Git commands I use 90% of the time and what they actually do in plain English
- How to think about branches and pull requests in the context of a dbt project

### 19. My First Time Reviewing a Pull Request for a SQL Model
- What to actually look for when reviewing a dbt PR: logic, grain, tests, documentation, naming
- How code review changed the way I write models knowing someone else will read them

### 20. How I Set Up a Local dbt + DuckDB Project for Free to Practice Every Day
- Step-by-step setup: dbt Core, DuckDB, and a CSV dataset loaded and ready to model
- Why this free local setup is all you need to build a strong portfolio

### 21. Orchestration Basics: What Airflow and dbt Cloud Are Actually Doing
- The concept of a DAG and how orchestration tools decide what runs when and in what order
- How dbt Cloud's scheduler differs from a full orchestrator like Airflow and when each makes sense

### 22. What I Learned Reading 10 Real Analytics Engineer Job Descriptions
- The tools and skills that appeared in almost every posting (and the ones that only appeared once)
- What "analytics engineer" actually means across different company sizes and data maturity levels

---

## Month 5–6 — Capstone & Job Search

### 23. I Built a Full Analytics Project From Raw Data to Dashboard — Here's What I Learned
- End-to-end walkthrough: raw IPEDS data → dbt models → a reporting layer → a simple dashboard
- The decisions I had to make along the way that no tutorial prepares you for

### 24. How My Higher Ed Background Is Actually an Advantage in the Analytics Engineer Job Hunt
- Domain expertise is underrated: I understand the business logic behind the data, not just the data itself
- Industries that have similar data complexity (healthcare, government, nonprofits) where I can target roles

### 25. The Portfolio Project That Got Me Interviews as a Career Changer
- What's in my GitHub and why I organized it the way I did
- The README template I use so that a hiring manager can understand the project in under 2 minutes

### 26. Questions I Asked in Every Analytics Engineer Interview (And What I Learned From the Answers)
- The questions that revealed the most about data culture and team maturity at each company
- Red flags and green flags I noticed in how teams talked about their data stack and modeling practices

---

## Tips to Keep in Mind Across All Posts

- **Use one consistent dataset throughout** — IPEDS (federal higher ed data) is free, public, and plays directly to your domain expertise. Hiring managers will notice the continuity.
- **Show your reasoning, not just your code** — explain *why* you made a decision, not just what you did. This is what separates strong AE candidates from people who just followed a tutorial.
- **Be honest about what confused you** — posts about mistakes and misconceptions get more engagement and are more useful to readers who are learning alongside you.
- **Link your posts to your GitHub** — every post that includes code should point to a real repo so readers (and recruiters) can see the full project.
