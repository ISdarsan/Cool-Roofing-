# Cool Roofing — Pre-Modelling Worksheet

**Data science is not just about applying tools or statistics.**
It is about understanding *where* a solution fits and *what value* it brings. You are not here to
drop an algorithm on top of a spreadsheet. You are here to understand what each person in this
story actually wants — the customer, and the company — and to build something that serves them.

Fill this worksheet in your own words *before* writing a single line of model code. It is a compass,
not an exam. A seasoned practitioner does this in their head; you will do it on paper.

---

## Section 0 — Understand the product (2-3 lines each)
1. In plain words, what is a cool roof and how does it work?
2. What changes for a building after it gets one? (list the physical effects)
3. Where in India / what kind of building would benefit most? Why?

---

## Section 1 — The customer's mind
*A person is thinking about spending money on their roof.*

1. What 2-3 questions is the customer most anxious to get answered?
2. Success, from the customer's side:
   2.1 What would a *good outcome* look like to them? (comfort, bill savings, cost, …)
   2.2 What metric(s) could the customer actually *look at* afterwards to say "yes, this worked"?
3. What would make them regret the decision?
4. What would make them tell a neighbour about it?

---

## Section 2 — The company's mind
*The company wants to grow and stay profitable.*

1. What does the company want out of every installation? (list 3-4)
2. How does a *happy* customer help the company beyond this one sale?
3. Some customers are good business; some become a headache. **How can the company tell — beforehand?**
   What could it know at (or before) install time that hints at which kind it is?
4. Which single roofing method should the company push, if any? What would you need to know to answer that?

---

## Section 3 — The walk-in (role play)
*A customer walks in; a sales rep sits across the table.*

Write a short dialogue (about 6-10 back-and-forth exchanges) between the **walk-in customer** and the
**sales representative**. The customer asks what they care about; the rep asks the questions they need
to give an answer. Write it as a real conversation:

> **Customer:** …
> **Rep:** …

By the end of your role play, you should notice: *which facts can the rep learn at the table on day one,
and which can only be known after the work is done?*

---

## Section 4 — Turn wishes into predictions
Build this table yourself. We give you only the column names below — you fill every row.

| Who (company / customer) | What they want to know (e.g. cost, comfort) | Which column(s) answer it | Which column(s) can we predict / estimate it from |
|---|---|---|---|
| | | | |

**Dataset columns for reference:**
`building_id`, `location_type`, `building_size`, `building_age`, `cool_roofing_quality`,
`roofing_type`, `cool_roofing_method`, `cool_roofing_cost`, `avg_roof_temperature_before`,
`num_ac_units`, `avg_units_before`, `avg_roof_temperature_after`, `avg_units_after`,
`customer_given_rating`, `customer_referral`.

---

## Section 5 — Earn the data's trust
1. Name 3 columns where a value could be *impossible*, and what the impossible value would look like.
2. For each problem you find: will you **drop** it, **fix** it, or **fill** it? State your reason in one line.
3. One sentence: why can't we skip this step and just model the raw dump?

---

## Section 6 — Plan of attack (fill last)
For **each** of the five prediction problems, name a baseline and a step up from it:

| Prediction problem | Simplest approach (baseline) | A more advanced model you'd consider |
|---|---|---|
| Installation cost | | |
| Roof temperature drop | | |
| Monthly electricity units saved | | |
| High customer rating | | |
| Customer referral | | |

Then:
1. Which target will you model first, and why that one?
2. When you're done, what one recommendation will you hand the company?

