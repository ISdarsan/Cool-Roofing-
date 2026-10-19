# Cool Roofing Cost & Impact Predictor

## Background

A cool roofing company has been installing heat-reflective roofing across buildings in India for several years. Their internal server has gone through multiple revisions and migrations - data was entered by different teams, systems were swapped mid-way, and nobody cleaned up after themselves. What you have is a raw dump of 2,500 installation records.

## Dataset

Each row represents one building that received a cool roofing installation. The dataset captures:

- **Building profile** - location type (rural to dense urban), floor area, and age at the time of installation.
- **Roofing details** - the existing roof construction type, the cool roofing product and method chosen, and its quality grade.
- **Installation cost** - total cost of the intervention in INR.
- **Before/after measurements** - average roof surface temperature (°C) and average monthly electricity consumption (kWh), measured before and after installation.
- **Customer feedback** - a satisfaction rating (1-5) and whether the customer referred the service to others.

## Problem Statement

When a new customer approaches the company, a sales rep can ask a few simple questions about their building. Based on that, can you predict:

1. **How much will the installation cost?**
2. **How much will their roof temperature drop?**
3. **How many electricity units will they save per month?**
4. **How likely are they to give a high rating after installation?**
5. **How likely are they to refer the company to someone else?**

Before you model anything, the data needs to earn your trust. Columns may have impossible values, fields left blank, or records that simply don't make sense - find them, decide what to do, and document your choices.

A good solution will not just produce accurate predictions, but explain which factors drive cost and outcomes the most - and whether the company should recommend a particular roofing method to maximise both customer satisfaction and referrals.
