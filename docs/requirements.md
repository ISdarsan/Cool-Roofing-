# Requirements (as read from the organiser files)

## Five prediction tasks (problem statement)
| # | Question | Target | Required metrics |
|---|---|---|---|
| A | How much will the installation cost? | `cool_roofing_cost` (INR) | MAE, RMSE |
| B | How much will the roof temperature drop? | `avg_roof_temperature_before − avg_roof_temperature_after` (°C) | MAE, RMSE |
| C | How many electricity units will they save per month? | `avg_units_before − avg_units_after` (kWh/month) | MAE, RMSE |
| D | How likely is a high rating? | `customer_given_rating` 1–5 | Accuracy, MAE |
| E | How likely is a referral? | `customer_referral` | Accuracy, F1 |

Additional asks in the statement: *find and document impossible values / blanks / nonsensical records*, *explain which factors drive
cost and outcomes*, and *say whether the company should recommend a particular roofing method to maximise satisfaction and referrals*.

`approach.md` asks for a business-first framing: customer vs. company goals, what is knowable on day one vs. only after the work,
a wishes→predictions table, data-trust checks, and a baseline + a step-up model per problem.

## Ambiguities and how they are resolved
| Ambiguity | Resolution (all configurable / documented) |
|---|---|
| Evaluation-metrics image is referenced but absent | Metrics follow the brief; no extra metric is used for selection. |
| "High rating" is not defined | **≥ 4** (`config.HIGH_RATING_THRESHOLD`), labelled as an assumption everywhere it is shown. |
| Rating needs *Accuracy and MAE* (different formulations) | Numeric rating regression → MAE; separately trained high/not-high classifier → Accuracy (derived-from-regression accuracy also reported). |
| Units of `avg_units_*` | Problem statement says "average **monthly** electricity consumption (kWh)" → monthly kWh. |
| Unit of `building_size` | Not stated ("floor area"). Treated as an opaque area unit; UI says "area (sq ft assumed)". |
| Positive class for F1 | Referral = True. Binary F1 (not macro/weighted). |
| Does the prediction happen before installation? | Yes (sales rep, new customer) → only pre-installation facts are model inputs. |
| Electricity tariff | Not in the data → never assumed; user supplies INR/kWh for payback. |
