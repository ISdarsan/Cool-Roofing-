# Speaker notes: Cool Roofing Intelligence (15 slides, about 13 minutes)

Planned total: 13.00 minutes, leaving about 2 minutes of buffer in a 15-minute slot.

## Slide 1 (~0.5 min)
[~0.5 min] Good [morning/afternoon]. We built Cool Roofing Intelligence: a system that turns a cool-roofing company's historical installation records into estimates a sales representative can use before the first quote: what it will cost, how much cooler the roof gets, how many electricity units are saved, and how the customer is likely to feel afterwards. Everything you will see today comes from our saved evaluation reports and the running application, including the limitations.

Takeaway: predictive, explainable and honest decision support for cooler buildings.

Transition: let me start with the problem.

## Slide 2 (~0.75 min)
[~0.75 min] A customer walking in asks four things. How much will it cost? How much cooler will my roof get? How much electricity will I save each month? And will I be happy enough to recommend you? Today a sales rep answers these from experience. The company, however, has a valuable asset: records of thousands of past installations with the cost, the temperatures and electricity use before and after, and the customer's rating and referral. We have 2,500 of them. The opportunity is to learn from those records and give consistent, explained answers.

Takeaway: the data to answer the questions already exists.

Transition: here is the solution we built on it.

## Slide 3 (~0.75 min)
[~0.75 min] The product has three layers. The core, which is implemented and tested, answers the five required questions with six models, because customer rating is both a numeric prediction and a high-or-not classification. It starts with an audit and cleaning pipeline, because the raw data has problems. It compares every valid roofing option for the same building, explains each prediction, and produces a PDF proposal. The AI assistant is optional: it never calculates numbers itself, it only calls the same prediction service. The three items on the right are future ideas and are not built.

Takeaway: a complete decision-support system, with a clear line between what exists and what is planned.

Transition: all of it rests on the dataset, so let's look at that.

## Slide 4 (~0.75 min)
[~0.75 min] The dataset has 2,500 installation records and 15 columns: building profile (location, size, age), roofing details (existing roof, method, quality), cost, roof temperature and monthly electricity before and after, the customer rating from one to five, and whether the customer referred us. Cost is right-skewed and scales with building size. Roof temperature drops by about 2 to 13 degrees on valid records. Electricity savings are strongly tied to current usage; note the dashed line at zero, because some buildings used more electricity afterwards. Four input columns have 106 missing cells in total. About 36% of customers referred us, so referral is an imbalanced target.

Takeaway: rich data with skewed targets and some gaps.

Transition: and some values that should not be trusted.

## Slide 5 (~1.0 min)
[~1 min] Before modelling we audited every column. Two problems stood out. First, exactly 120 degrees appears as a roof temperature in 58 records, never anywhere near other readings, which would imply drops of plus or minus ninety degrees: that is confirmed invalid. Second, exactly zero kilowatt-hours appears 48 times while one to three kWh never occurs, and those rows are physically inconsistent. We treated them as placeholders. That is an inference, a strong one, but not a confirmed fact. We set invalid values to missing rather than guessing, and exclude a record only from the task that needs it. The result was measurable: electricity-savings error on the hold-out fell from 4.45 to 3.62 kWh. We also kept 167 buildings where electricity use went up, because those are real adverse outcomes, and deleting them would flatter the model. No record was deleted outright.

Takeaway: cleaning decisions are documented, evidence-based and measurable.

Transition: with clean data, we must also prevent leakage.

## Slide 6 (~1.0 min)
[~1 min] Two outcomes are derived: temperature drop is temperature before minus after, and electricity saved is units before minus after. The key design rule is that a model may only use facts known before installation. On the right, nine inputs are allowed: location, size, age, existing roof, the proposed method and quality, and three pre-installation measurements. Realised cost, the after-installation measurements, the customer's rating and the referral flag are forbidden, as is the building ID. This matters: for example, the referral rate is 61 percent for four-star-plus customers against 9 percent for one-to-two stars, so using the rating to predict referral would be pure leakage. For evaluation we fixed a hold-out of 500 buildings before any modelling. Tuning used five-fold cross-validation on the other 2,000. All preprocessing is fitted on training data only. Automated tests and independent controls, such as models trained on shuffled labels, confirm there is no leakage.

Takeaway: the evaluation answers 'how well will this work on a new building?'.

Transition: here is the system that implements it.

## Slide 7 (~1.0 min)
[~1 min] The pipeline flows left to right. The raw CSV is read-only; the audit and cleaning step produces a cleaned dataset and an audit trail; a feature contract and shared preprocessing pipelines feed training and evaluation, which saves six model files plus metadata. A FastAPI service loads the models once and serves prediction, option comparison, Shapley-value explanations, a payback calculator and PDF proposals. The React and TypeScript front end calls that API. The optional assistant also goes through the API: the language model can only call validated tools, so it never produces official numbers. Quality gates: 153 backend tests and 30 frontend tests including accessibility checks, a clean type check and production build.

Takeaway: one shared feature contract keeps training and serving consistent.

Transition: now the models themselves.

## Slide 8 (~1.0 min)
[~1 min] There are five prediction tasks but six trained models, because the rating task has two parts: a regression for the one-to-five rating, scored by mean absolute error, and a classifier for 'high rating', scored by accuracy. Regression predicts a number, such as rupees or degrees; classification predicts a category or probability, such as will the customer refer. Note that the best cost model is not the fanciest one: a log-linear ridge regression beat gradient boosting because cost behaves like size times a rate. The two LightGBM savings and cooling models predict a fraction and multiply it back, which suits how those outcomes scale. Each model is compared with a baseline and tuned with cross-validation only.

Takeaway: we chose models by validation evidence, not sophistication.

Transition: how do we measure them?

## Slide 9 (~1.0 min)
[~1 min] A quick guide to the metrics. MAE is the average size of the miss, in the unit of the thing we predict. RMSE also measures error but squares it first, so large misses count more. For yes-or-no tasks, accuracy is the share of correct answers, but it can be misleading when classes are imbalanced: always predicting 'no referral' is right 63 percent of the time. F1 balances precision, how many flagged customers truly are, with recall, how many true cases we catch. Our protocol: the 500-building hold-out was fixed before modelling and scored once. Baselines were declared before looking at it. We call a model 'better' only if a paired bootstrap interval of the improvement excludes zero. An independent review also rebuilt every number from the saved models and replicated the whole selection on a hold-out that exploration had never touched.

Takeaway: results are honest and statistically checked.

Transition: first, the three regression results.

## Slide 10 (~1.0 min)
[~1 min] For the three regression tasks, each chart compares the baseline, which just predicts the training average, with our selected model, in the task's own unit. Cost: the typical miss falls from ₹637,024 to ₹25,299, about 25 times smaller. Cooling: from 1.80 to 0.41 degrees Celsius. Electricity saved: from 38.4 to 3.62 kWh a month. In all three, the paired bootstrap interval of the improvement is far above zero. R-squared is 0.998, 0.948 and 0.987. An ablation shows these come from legitimate inputs: remove building size and the cost R-squared collapses to 0.22; remove method and quality and cooling drops to 0.25. One caveat: our nominal 80 percent likely-range intervals actually cover 78%, 79% and 76%, so they are slightly narrow.

Takeaway: strong, statistically supported accuracy where the data is structured.

Transition: the human-outcome tasks are harder.

## Slide 11 (~1.0 min)
[~1 min] These two tasks depend on what happens after installation, so they are harder at quote time. For ratings, mean error is 0.77 stars against 0.92 for the baseline, a significant gain. For high rating, defined as four stars or more, which is our assumption, accuracy is 0.688 against 0.576 for the majority class, also significant, and the AUC is 0.76. But the F1 of 0.61 is not distinguishable from simply saying everyone will rate highly. Referral is the weakest. At its best-F1 cut-off the model catches most referrers, recall 0.86, but flags many people wrongly. F1 is 0.57, statistically indistinguishable from predicting everyone refers at 0.54, and accuracy 0.512 is significantly worse than the majority-class 0.626. The AUC of 0.66 shows a real but weak ranking signal. Even knowing the realised savings would only reach 0.68 AUC, so the information mostly isn't in the data. We use it to rank follow-ups, never to promise.

Takeaway: being transparent about weak results is part of the engineering.

Transition: what actually drives the strong results?

## Slide 12 (~1.0 min)
[~1 min] On the left is global feature importance from the hold-out set, meaning how much accuracy drops when an input is shuffled. Building size dominates cost; the roofing method and quality dominate cooling; current electricity use and AC count dominate savings; AC count and the method matter most for ratings and referrals. On the right, we ask the model what happens on concrete roofs, the only roofs that accept all five methods, if every building used each method in turn. Coated options are consistently strongest on cooling, savings and predicted satisfaction, but there is no single best method for everyone, and roof type limits what can be installed. Basic coating is cheapest but weakest. These are associations from past installations, not proof of cause. Also, at a plausible electricity price the savings alone would take very long to repay the cost, so the business case rests on comfort and benefits the data doesn't measure.

Takeaway: explanations are specific, computed, and carefully worded.

Transition: and here is how it looks in the product.

## Slide 13 (~1.0 min)
[~1 min] These are real screenshots from the running app. Top left, the prediction workspace: the rep enters a few facts and gets cost, cooling, electricity savings, rating and referral, each with a likely range and a reliability flag, plus a 'why this estimate' explanation and an optional payback calculator that never assumes a tariff. Top right, the comparison page ranks every valid method and quality combination for the same building, with adjustable priorities and a transparent score. Below: the model performance page with baselines and confidence intervals, the PDF proposal page, and the optional AI assistant, which answered live through Gemini by calling the same prediction tools. The Claude integration is built, but we could not live-test it because that account has no credit.

Takeaway: this is working software, not a mock-up.

Transition: finally, value and limits.

## Slide 14 (~0.75 min)
[~0.75 min] For the business, the product gives a consistent first estimate in well under a second, lets a rep compare fifteen option combinations for the same building, explains each prediction, makes every assumption visible, and produces a proposal PDF. We are equally clear about limits. Predictions are estimates, not guarantees. The zero-kilowatt-hour treatment is an inference. Rating and referral models are weak. The high-rating threshold is our assumption. Method effects are associations from observational data, there is no time or location field to test drift, and the Claude integration has not been live-tested. The roadmap on the right is future work only: a digital-twin simulator, a carbon-savings tracker and geo-climate intelligence are not built. The most valuable next step is validating against newly measured installations.

Takeaway: real value, bounded claims.

Transition: to close.

## Slide 15 (~0.5 min)
[~0.5 min] To close. We built a complete system: a cleaned and audited dataset, six models for five tasks, honest hold-out evaluation with baselines and significance tests, explanations, an option-comparison engine, PDF proposals and an optional assistant. The evaluation supports strong predictions of cost, cooling and electricity savings, and modest but real signal for ratings, while referrals remain a ranking aid only. Our strongest contribution may be the discipline: finding the hidden placeholders, preventing leakage, and reporting weak results plainly. The next step is validating on new, measured installations. Thank you; we're happy to take questions.

Takeaway: trustworthy, explainable decisions from historical roofing data.
