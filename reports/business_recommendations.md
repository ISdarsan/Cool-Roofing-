# Business Recommendations

_Written from the computed results in `business_insights.md`; every number is traceable to the trained models and the real data._

## Headline answer: which roofing method should the company recommend?
**There is no single best method for every customer, but the coated products are consistently stronger on the outcomes customers notice.**

- On concrete roofs, **tiles + coating** has the highest predicted satisfaction (58% chance of a high rating) and referral (48%), with a 8.2 °C average temperature drop and 65 kWh/month saved, at an average ₹1,335,396.
- **sheet + coating** is close behind (55% high rating, 47% referral) and about 6% cheaper than tiles + coating (₹1,251,011 vs ₹1,335,396).
- **basic coating** is the cheapest (₹893,144) but gives the weakest outcomes (4.6 °C, 35% high rating, 32% referral) and the worst cost per °C (₹192,352 vs ₹161,357 for sheet + coating), so it is not good value despite the low ticket price.
- Method must match the existing roof: tile products need tile or concrete roofs; sheet products need sheet or concrete roofs. By roof type (highest combined chance of a high rating and a referral):
  - **concrete_flat**: tiles + coating (P(high rating) 58%, referral 48%, avg ₹1,316,506); cheapest is basic coating (₹880,509).
  - **concrete_slanted**: tiles + coating (P(high rating) 57%, referral 48%, avg ₹1,363,839); cheapest is basic coating (₹912,167).
  - **sheet**: sheet + coating (P(high rating) 50%, referral 39%, avg ₹1,011,699); cheapest is basic coating (₹722,289).
  - **tiles**: tiles + coating (P(high rating) 45%, referral 39%, avg ₹1,522,755); cheapest is basic coating (₹1,018,454).
- Practical rule: **lead with the strongest option that fits the roof** (above), then use the comparison tool to confirm budget fit.

## Quality grade
High quality costs about ₹570,457 more than basic on average but is associated with +3.0 °C more cooling, +18 kWh/month and a 54% (vs 32%) chance of a high rating. Recommend high or intermediate quality to customers whose priority is comfort or satisfaction; basic quality only where budget is binding.

## Which customers to prioritise (knowable at the table)
- Customers with 3 AC units show a 52% high-rating rate and 41% referral rate; customers with no AC show 17% and 21%, and their average saving is essentially nil (-0.0 kWh/month).
- Set expectations honestly with low-usage, no-AC buildings: little electricity saving is possible (a minority even see a small increase), although roof temperature still falls. Lead with comfort and roof-life messages rather than bill savings for them.
- Current electricity use, AC count and starting roof temperature are the strongest day-one signals; they are the questions to ask first.

## Economics: be upfront about payback
- Predicted electricity savings are modest relative to installation cost. For a 10-year payback from electricity alone the tariff would need to be at least ₹147/kWh even for the best-value option (sheet, basic). Use the app's payback calculator with the customer's real tariff; do not sell this on bill savings alone.
- The value case rests on cooling (roof temperature drop) and on benefits the data does not measure; position the product accordingly.

## What not to promise
- Referral is only weakly predictable (hold-out ROC-AUC 0.66); use it to rank follow-ups, never to forecast individual customers.
- Payback depends on a tariff the data does not contain; the app computes it only from a tariff the user enters and labels it an assumption.
- Savings and satisfaction are historical associations; they are not guarantees for a new building.

## Limits of this evidence
- Observational data: customers who chose a method may differ in unrecorded ways.
- No time or location identifiers, so drift and regional effects cannot be checked.
- Temperature, savings and cost models are strong (see `model_evaluation.md`); rating and referral models carry real but modest signal.
