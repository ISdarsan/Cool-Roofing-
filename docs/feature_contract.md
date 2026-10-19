# Feature Contract & Leakage Prevention

The contract lives in one place — [`src/coolroof/config.py`](../src/coolroof/config.py) — and is consumed by training, inference, the API schemas and the tests.

## Rule
A feature is allowed **only if a sales rep can know it before installation**. Everything that exists only after the work (or that *is* a target) is forbidden for every task.

| Allowed pre-installation inputs (9) | Type | Notes |
|---|---|---|
| `location_type` | category | rural / intermediate / urban / dense_urban |
| `building_size` | number | floor area; unit not stated in the source |
| `building_age` | number | optional at prediction time (imputed + flagged) |
| `roofing_type` | category | existing roof; constrains valid methods |
| `cool_roofing_method` | category | **decision variable** (proposed by the rep) |
| `cool_roofing_quality` | category | **decision variable** |
| `avg_roof_temperature_before` | number | optional (imputed + flagged) |
| `num_ac_units` | number | optional |
| `avg_units_before` | number | optional; monthly kWh |

One derived, input-only feature is added inside the pipeline: `log_building_size`.

| Forbidden for **all** tasks | Why |
|---|---|
| `building_id` | record number, no business meaning |
| `cool_roofing_cost` | realised cost is only known after the quote; also the Task A target. Excluded even for rating/referral because the prediction happens before installation |
| `avg_roof_temperature_after`, `avg_units_after` | post-installation measurements; only used to *build* the Task B/C targets |
| `customer_given_rating`, `customer_referral` | post-installation outcomes (Tasks D/E targets) |
| `temperature_drop`, `units_saved`, `high_rating` | derived targets |

## Per-task contract
| Task | Target | Features | Eligible rows |
|---|---|---|---|
| A cost | `cool_roofing_cost` | the 9 inputs | valid cost **and** building size present |
| B temperature drop | `before − after` | the 9 inputs | both temperatures valid (not the 120 °C placeholder) |
| C units saved | `avg_units_before − avg_units_after` | the 9 inputs | both kWh readings valid (not the 0 placeholder) |
| D rating / high rating | rating 1–5 / `rating ≥ 4` | the 9 inputs | valid rating |
| E referral | `customer_referral` (True = positive) | the 9 inputs | valid flag |

## How it is enforced
1. **Central config** — `PRE_INSTALLATION_FEATURES`, `FORBIDDEN_AS_FEATURES`, `TASKS[...]`.
2. **`validate_feature_contract(task)`** (runs at the start of training): fails if a task lists a forbidden column, its own target, or anything off the allow-list.
3. **`assert_frame_has_no_leakage(X)`** runs inside `design_matrix`, immediately before any model sees data.
4. **Pipelines** (`FeatureEngineer → imputer/encoder/scaler → model`) fit all preprocessing on training folds only. A test proves imputer statistics equal the *training* medians, not the full-data medians.
5. **API**: request models use `extra="forbid"`, so a client that sends `cool_roofing_cost`, `customer_given_rating`, `avg_units_after`… gets HTTP 422 instead of being silently ignored.
6. **Tests**: `tests/test_features_leakage.py` (contract violations rejected, schema agreement) and `tests/test_api.py::test_post_installation_fields_are_rejected`.
7. **Hold-out hygiene**: the 20 % test buildings are fixed before modelling; a test asserts none of them appears in any training/OOF frame.

## Prediction-time schema
Machine-readable via `GET /api/schema` and `coolroof.features.input_schema()`. Required: location, size, existing roof, method, quality. Optional (imputed with training-fold statistics and reported as a warning): age, starting temperature, AC units, current kWh. Valid method/roof pairs come from the data (a tile product on a sheet roof, or vice versa, never occurs and is rejected with HTTP 422).
