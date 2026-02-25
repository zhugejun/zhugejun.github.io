---
title: 'The Tale of Oakwood Motors: A Credit Risk Story'
date: '2026-02-24'
categories:
  - Risk Analysis
tags:
  - credit-risk
  - storytelling
  - auto-finance
---

# The Tale of Oakwood Motors: A Credit Risk Story

---

## Chapter 1: The Dealership Door

Maria had always loved cars. So when she walked into **Oakwood GM Dealership** on a crisp Saturday morning and pointed at the midnight-blue Chevy Silverado on the showroom floor, her heart was already set.

"Let's get you into that truck," said the dealer, grinning. He sat Maria down, pulled up the finance portal, and submitted her application to **Great Mountain Financial (GMF)** — the captive lender for GM vehicles.

And just like that, the **origination pipeline** kicked into gear.

Maria's FICO score was pulled. Her income was verified. Her application flew across a fiber optic cable to GMF's credit decisioning engine, where — in a matter of seconds — she would be judged by the **Five C's of Credit**.

---

## Chapter 2: The Five Judges

Deep inside GMF's system, Maria's application appeared before five invisible judges. Each one had a single question.

**Judge Character** looked at Maria's credit history. "FICO score: 710. No prior collections. One late payment from three years ago, but otherwise clean." He nodded approvingly. _Character is about trustworthiness — can you be counted on to pay?_ Maria's history said yes.

**Judge Capacity** pulled up her income. Maria earned \$5,000 a month. The Silverado's payment would be \$680. He did the math quickly: $680 ÷ $5,000 = **13.6% PTI (Payment-to-Income)**. "She can afford this," he said. _Capacity asks: even if you're trustworthy, do you actually have the money?_ If her payment had been $1,400 — a 28% PTI — he would've slammed the gavel down.

**Judge Capital** looked at her down payment. Maria had saved \$6,000 to put down on the \$38,000 truck. That meant she was financing \$32,000 on a $38,000 vehicle — an **LTV (Loan-to-Value) of 84%**. "Skin in the game," he said with satisfaction. _Capital is about how much the borrower has invested._ If Maria had put zero down, her LTV would be 100% — and she'd start the loan underwater the moment the truck depreciated.

**Judge Collateral** inspected the truck itself. A brand-new Silverado? Strong resale value. It holds its price well at auction. "Solid collateral," she ruled. _The collateral is the safety net — if everything goes wrong, can we sell this and recover our money?_ A seven-year-old sedan with 120,000 miles would've been a different story.

**Judge Conditions** looked at the broader picture. Interest rates were moderate. Unemployment was low. Maria wanted a 60-month term — not the risky 84-month variety. "Conditions are favorable," he declared. _Conditions account for the world outside the borrower — the economy, the rate environment, the loan structure itself._

All five judges agreed. Maria was approved. Tier: **Near-Prime**. Rate: 6.9%. Term: 60 months.

GMF purchased the contract from the dealership, wired the dealer their money, and Maria drove her Silverado home that evening, grinning in the rearview mirror.

---

## Chapter 3: The Two Paths — Loan vs. Lease

Maria chose a **loan** — a retail installment contract. She was buying the truck outright and building equity with every payment. In 60 months, it would be hers, free and clear.

But her coworker **Derek** took a different path. He walked into the same dealership and **leased** a Chevy Blazer EV. He didn't own it — GMF did. Derek was essentially renting it for 36 months, paying for the vehicle's depreciation during that time.

Here's where their risks diverge:

**Maria's risk (the loan):** If she loses her job and stops paying, GMF repossesses the truck and sells it at auction. The danger is **negative equity** — if her loan balance is higher than what the truck is worth, GMF eats the difference. That gap is called **loss severity**.

**Derek's risk (the lease):** Even if Derek pays perfectly every month, GMF still faces **residual value risk**. They projected the Blazer EV would be worth \$25,000 when Derek returns it in three years. But what if battery technology leaps forward, a sleeker model hits the market, and used Blazer EVs are suddenly only worth \$19,000? That's a \$6,000 loss per vehicle — and Derek didn't miss a single payment. Multiply that by 50,000 leases and you're staring at a **$300 million hole**.

This is why leasing is a different animal. Loans are about _borrower behavior_. Leases add _market risk_ on top of it.

---

## Chapter 4: The Check Engine Light — Leading vs. Lagging Indicators

Six months after Maria's loan was funded, a new analyst named **Priya** joined GMF's credit risk team. Her manager, Hector, sat her down on Day One with a piece of advice she'd never forget.

"There are two kinds of signals in this business," Hector said. "**Check engine lights** and **autopsy reports**. You want to get really good at reading the check engine lights."

He pulled up the portfolio dashboard.

"See this? **30+ DPD delinquency rate: 4.0%**. Three months ago it was 3.2%." He leaned back. "That's a check engine light. It's a **leading indicator**. The damage hasn't happened yet, but it's coming."

He clicked into the next metric. "**Roll rates**: 40% of accounts that were 30 days past due last month have now rolled to 60 DPD. Last year's average was 30%. That means borrowers are _not_ catching up on their payments. They're sinking deeper."

"And this," he pointed to a third number, "**early payment defaults** — loans going delinquent within the first six months of origination. If this spikes, it means something went wrong at underwriting. We approved people who couldn't afford the car from Day One."

Priya scribbled furiously. "So those are the early warnings. What about the autopsy reports?"

Hector smiled. "Those are the **lagging indicators**. **Net charge-off rate**: we wrote off \$200 million last quarter and recovered \$80 million at auction. On a $50 billion portfolio, that's an annualized NCO of about 0.96%. That tells us how we did — past tense. The borrowers already defaulted. The trucks already went to auction. There's nothing left to prevent."

"Then there's **loss severity** — how much we actually lost per defaulted loan. Borrower defaults with a \$28,000 balance, we repo the truck, sell it for \$18,000, subtract \$2,000 in costs. We recovered \$16,000 and lost $12,000. That's 42.9% severity."

"And finally, **recovery rates** — the flip side of severity. How much of what we're owed do we actually get back?"

He turned to Priya. "Leading indicators are where you earn your paycheck. By the time lagging indicators tell you something, it's too late to change the outcome. You can only learn from it."

---

## Chapter 5: The Vintage Mystery

A few weeks into the job, Priya noticed something strange. The overall portfolio delinquency rate looked stable — hovering around 3.8%. But Hector had taught her to never trust the average.

"Dig deeper," she muttered, and pulled up a **vintage analysis** — grouping loans by the quarter they were originated and tracking each group's performance over time.

The picture changed completely:

| Vintage | Cumulative Loss at 12 Months | Avg FICO at Origination |
| ------- | ---------------------------- | ----------------------- |
| Q1 2024 | 0.8%                         | 705                     |
| Q3 2024 | 1.4%                         | 680                     |
| Q1 2025 | 0.7%                         | 715                     |

The **Q3 2024 vintage was rotting**. Losses were nearly double the Q1 2024 cohort — and the average FICO was 25 points lower. That wasn't a recession. That wasn't bad luck. That was a **credit policy problem**. Someone had loosened the underwriting standards during Q3, approving riskier borrowers, and now the consequences were showing up.

But the overall portfolio number had masked it, because the healthy Q1 2024 and Q1 2025 vintages were diluting the damage.

Priya understood: _vintage analysis is like tracking each graduating class separately._ If you just look at the overall graduation rate of the whole university, you'd never know that the Class of 2024 had a dropout problem while the Class of 2025 was thriving. You have to isolate each cohort by entry date and track them at consistent intervals to see what actually changed.

She wrote up her findings and flagged it to Hector. "Q3 2024 is our problem child."

---

## Chapter 6: The Great Debate — Tighten or Loosen?

Priya's analysis landed on the desk of GMF's Chief Risk Officer, who called a meeting that would shape the company's direction for the next year. The room was split into two camps.

**The Risk Hawks** wanted to tighten. "Raise the minimum FICO from 620 to 660," they argued. "The Q3 2024 vintage proves we went too loose. Cut off the riskiest borrowers and our delinquency rates will improve within a year."

They were right — but there was a cost. Tightening would eliminate about 12% of applicants. The **approval rate** would drop. Fewer loans meant less interest income.

**The Growth Doves** pushed back. "You can't just look at risk in isolation. Our **penetration rate** is at 39% — GMF finances 39% of all GM vehicle sales. If we tighten too much, that number drops. Dealers start sending applications to competing lenders. GM sells fewer cars. The parent company won't be happy."

And there it was: the **fundamental credit policy trade-off**. Tighten policy and you get better loan quality but less volume, weaker dealer relationships, and lower penetration. Loosen policy and you get growth, happy dealers, and more interest income — but the vintages might blow up in 12 to 24 months.

The CRO turned to Priya's team. "Don't tell me whether to tighten or loosen. **Quantify the trade-off.** <u>If we raise the floor to 660, exactly how many borrowers do we lose? What's the expected reduction in losses? What's the revenue impact? What does it do to penetration? Give me the numbers so I can make a decision."</u>

That, Priya realized, was the real job of a risk analyst. Not to say "be careful" — anyone can do that. The job is to **mine the data, analyze the impact, segment and monitor by region, dealer, tier, and vintage, then make a clear recommendation and present it to management with the "so what" front and center.**

---

## Chapter 7: The Numbers Tell the Story

Priya spent two weeks buried in SQL queries and SAS models. She segmented the marginal borrowers — the ones between FICO 620 and 659 who would be cut off by the proposed policy change. She built projections.

Her findings told a nuanced story:

Raising the floor to 660 would improve the 12-month delinquency rate on new originations by about 0.8 percentage points. But it would drop penetration from 39% to roughly 35%, and dealer complaints would increase. The lost interest income from those eliminated borrowers was about \$45 million annually — partially offset by roughly $30 million in avoided losses.

The net cost of tightening: about $15 million in lost revenue, but with a significantly healthier portfolio and lower reserve requirements.

She also proposed a middle path: keep the FICO floor at 620, but cap the **LTV at 100%** for anyone below 660 and limit their **term to 60 months max**. This would keep marginal borrowers in the pipeline but prevent the worst-case scenarios — the 117% LTV, 84-month loans that create negative equity traps nobody can escape.

Hector reviewed her presentation. "Lead with the 'so what,'" he reminded her.

So she did. Slide one: _"Q3 2024 policy loosening is projected to cost $38M in excess losses. Here are three options to course-correct, with trade-offs quantified."_

Management chose the middle path.

---

## Epilogue: Maria's Payment

Back in the real world, Maria made her 12th payment on the Silverado. She was on time, every time. Her loan balance was down to \$27,400, and the truck was still worth about $31,000. She had positive equity — the best position a borrower can be in.

Somewhere in GMF's system, her account was one of 500,000 being monitored by people like Priya — analysts who understood that behind every metric was a person, and behind every person was a number that told a story about risk.

Maria didn't know any of this. She just knew she loved her truck.

And that was fine. The system was working the way it was supposed to.

---

## Quick Concept Recap

| Story Moment                         | Concept                                                                                                                         |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Maria walks into the dealership      | **Origination pipeline** (application → decisioning → funding → servicing)                                                      |
| The Five Judges evaluate her         | **5 C's**: Character (FICO), Capacity (PTI), Capital (down payment/LTV), Collateral (the vehicle), Conditions (economy/term)    |
| Maria vs. Derek                      | **Loan vs. Lease** — borrower default risk vs. residual value risk                                                              |
| Hector's "check engine light" speech | **Leading indicators** (DPD, roll rates, early payment defaults) vs. **Lagging indicators** (NCO, loss severity, recovery rate) |
| Priya finds the Q3 problem           | **Vintage analysis** — isolating cohorts by origination date                                                                    |
| The Hawks vs. Doves debate           | **Credit policy trade-off** — tighten (better quality, less volume) vs. loosen (more volume, higher risk)                       |
| The CRO's challenge                  | **Approval rate** and **Penetration rate** as business KPIs                                                                     |
| Priya's final presentation           | **The analyst's role**: mine data → analyze impact → segment & monitor → recommend → present with the "so what"                 |
