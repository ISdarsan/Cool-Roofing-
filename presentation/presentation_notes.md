# Speaker notes: Cool Roofing Intelligence (8 slides)

Planned speaking time: 12.50 minutes (target 12-13), leaving buffer in a 15-minute slot.

## Slide 1 (~1.25 min)
[~1.25 min] Good [morning/afternoon]. Our project is Cool Roofing Intelligence. Cool roofing means heat-reflective roofing: coatings, sheets or tiles that keep the roof surface cooler, which should make the building more comfortable and cut cooling electricity. In our data, buildings across India that received these roofs cooled by about 2 to 14 degrees Celsius. The business problem is that a company wants to tell a new customer, before the quote, what it will cost and what they will get. Our objective is to turn historical installation records into that answer. In one sentence: we built a tested web system that predicts cost, cooling, electricity savings, rating and referral from 2,500 past installations, compares roofing options, and explains every estimate.

Takeaway: predictive, explainable decision support for cool roofing.

Transition: let me state the five problems precisely.

## Slide 2 (~1.5 min)
[~1.5 min] The organisers' problem statement asks for five predictions for a new building. One: the installation cost in rupees. Two: how much the roof temperature drops, which we compute as temperature before minus after. Three: the monthly electricity saved in kilowatt-hours, units before minus units after. Four: the customer's rating from one to five. Five: whether the customer will refer the company. Why does historical data help? Each past installation records what was known at quote time, such as building size, roof type, method and quality, and what actually happened afterwards. A model can learn that pattern and give a consistent, explained estimate instead of relying on experience alone. We also had to check whether the data could be trusted, which I will show next.

Takeaway: five well-defined prediction tasks, supported by real historical outcomes.

Transition: first, the dataset.

## Slide 3 (~2 min)
[~2 min] The dataset has 2,500 records and 15 columns. Inputs known before installation are location, size, age, existing roof, proposed method and quality, starting roof temperature, number of air conditioners and current monthly electricity use. The outcomes are cost, temperatures and electricity after installation, the rating and the referral. The bottom charts show the exploration: cost is right-skewed and scales with size; valid cooling is between about two and thirteen degrees; and electricity savings track current usage, with some buildings at or below zero. The audit found two silent problems. Exactly 120 degrees appears as a roof temperature in 58 records, far from every other reading, so it is invalid. Exactly zero kilowatt-hours appears 48 times while one to three never occurs, and those rows are physically inconsistent, so we treated them as placeholders. That second one is an inference, a strong one, not a confirmed fact. We set both to missing instead of guessing, and exclude a record only from the task that needs that value; nothing was deleted outright. We also kept 167 buildings whose electricity use actually went up, because those are real adverse outcomes. The cleaning had a measurable effect: electricity-savings error on unseen data fell from 4.45 to 3.62 kilowatt-hours. There are also 106 missing input cells, which are filled using training data only.

Takeaway: the data was audited, and every cleaning decision is documented and evidence-based.

Transition: next, the models.

## Slide 4 (~1.25 min)
[~1.25 min] There are five prediction tasks and six trained models, because customer rating is treated two ways: a numerical rating, and a high-or-not classification. We use regression when the answer is a number, such as rupees, degrees or kilowatt-hours, and classification when the answer is a category, such as will the customer refer, or is the rating high. For cost we used a log-linear ridge regression, because cost behaves like size times a rate, and that simple model actually beat gradient boosting. For cooling and savings we used LightGBM, a gradient-boosted tree model, to predict a fraction and multiply it back by the starting temperature or usage. Rating and high rating use LightGBM too, and referral uses logistic regression. Every model was compared against baselines and simpler alternatives using cross-validation only, never the test set.

Takeaway: simple models where the data is simple, with every choice backed by validation.

Transition: and the technology that delivers them.

## Slide 5 (~1.25 min)
[~1.25 min] The data flows left to right. The raw CSV is read-only and processed with pandas. The audit and cleaning step produces a cleaned dataset and an audit trail. Training and evaluation use scikit-learn and LightGBM, and save six model files with metadata. A FastAPI service loads those models once and serves predictions, option comparison, explanations and PDF proposals generated with ReportLab. The React, TypeScript and Vite front end, styled with Tailwind and charted with Recharts, calls that API. There is also an optional AI assistant. It has a free built-in mode, and a Gemini integration that we tested live. A Claude integration is built but we could not live-test it, because that account has no credit. In every mode the language model can only call our validated tools, so it never invents the numbers. For quality, we have 153 backend tests and 30 frontend tests including accessibility checks, a clean type check and a production build.

Takeaway: a conventional, maintainable stack with one shared feature contract between training and serving.

Transition: now the results.

## Slide 6 (~2.5 min)
[~2.5 min] First the metrics. MAE, mean absolute error, is the average size of the miss in the unit being predicted. RMSE also measures error but squares it, so large misses count more. R-squared is the share of variation explained. For yes-or-no tasks, accuracy is the share of correct answers and F1 balances precision and recall; ROC-AUC measures how well the model ranks cases, where 0.5 is chance. All results are on 500 buildings that were set aside before modelling and never used for training or tuning. Each chart has its own unit and scale. Cost: the typical miss is ₹25,299 against ₹637,024 for a baseline that just predicts the average. Cooling: 0.41 against 1.80 degrees. Electricity saved: 3.62 against 38.4 kilowatt-hours. R-squared is 0.998, 0.948 and 0.987. Rating and referral are weaker. Rating error is 0.77 stars against 0.92. High-rating accuracy is 68.8% against 57.6% for always guessing the majority, which is a significant gain, but its F1 is not distinguishable from a trivial rule. For referral, F1 is 0.569, statistically indistinguishable from predicting everyone refers at 0.544, and accuracy of 51.2% is significantly worse than the majority-class 62.6%. The AUC of 0.66 shows a real but weak ranking signal. These two outcomes depend on what happens after installation, which isn't available at quote time, so we use them only to rank, never to promise.

Takeaway: strong where the data is structured, honest where it isn't.

Transition: how it looks in practice.

## Slide 7 (~1.75 min)
[~1.75 min] These are real screenshots from the running application. Top left, customer prediction: a rep enters a few facts about the building and gets cost, cooling, electricity savings, rating and referral, each with a likely range and a reliability flag, plus a 'why this estimate' explanation. Next, the roofing-option comparison ranks every valid method and quality for the same building using a transparent, adjustable score. Third, the model performance dashboard, with baselines, confidence intervals and a confusion matrix. On the bottom row, the data-quality insights page shows the audit, then the PDF proposal page, built from real predictions, and the optional AI assistant. The assistant answered live through Gemini by calling the same prediction tools; the Claude integration is built but not live-tested. The business value is simple: a company can prepare a preliminary estimate in seconds, compare options before quoting, and hand the customer a proposal that states its assumptions. These are estimates, not guarantees, and we say so in the product.

Takeaway: this is working software that supports quoting.

Transition: to wrap up.

## Slide 8 (~1 min)
[~1 min] To conclude. The project turns 2,500 historical installations into a working system: an audited dataset, six models for five tasks, honest evaluation on unseen buildings, explanations, option comparison and PDF proposals. The strongest results are cost with an average miss of ₹25,299, cooling within 0.4 degrees and electricity savings within 3.6 kilowatt-hours. The value for a roofing company is faster, better-informed quotes. The limits are real: the dataset has no time or location fields, the zero-kilowatt-hour treatment is an inference, rating and referral are weak, the four-star threshold is our assumption, and predictions are estimates, not guarantees. Future scope, none of it built yet: a digital-twin simulator, a carbon-savings tracker, geo-climate intelligence, and above all validation using newly measured installations. Thank you, we're happy to take questions.

Takeaway: trustworthy, explainable decisions from historical data.
