# Data Dictionary

Source: `dataset/cool_roofing_data.csv` — 2,500 rows × 15 columns (verified). One row = one building that received a cool-roofing installation.

| Column | Type | Meaning | Observed values / range | Known when quoting? | Role |
|---|---|---|---|---|---|
| `building_id` | int | Record number 1…2500 (unique, sequential) | 1–2500 | n/a | identifier only — **never a feature** |
| `location_type` | category | Urbanisation level | rural, intermediate, urban, dense_urban (24 missing) | yes | feature |
| `building_size` | float | Floor area (unit not stated) | 300–3000 (28 missing) | yes | feature |
| `building_age` | float | Years at installation | 1–50 (23 missing) | yes | feature |
| `cool_roofing_quality` | category | Product quality grade | basic, intermediate, high | **proposed by rep** | feature / decision variable |
| `roofing_type` | category | Existing roof construction | concrete_flat, concrete_slanted, sheet, tiles | yes | feature |
| `cool_roofing_method` | category | Cool-roofing product/method | basic_coating, sheet, sheet_with_coating, tiles, tiles_with_coating | **proposed by rep** | feature / decision variable |
| `cool_roofing_cost` | int | Total installation cost, INR | 55,300–5,046,600 | no (it is the Task A target) | target A; forbidden elsewhere |
| `avg_roof_temperature_before` | float | Avg roof-surface temperature before, °C | 22.98–120 (32 sentinel = 120) | yes (survey/measurement) | feature |
| `num_ac_units` | float | Air-conditioning units in the building | 0–3 (31 missing) | yes | feature |
| `avg_units_before` | int | Avg monthly electricity before, kWh | 0–500 | yes (past bills) | feature |
| `avg_roof_temperature_after` | float | Avg roof-surface temperature after, °C | 17.81–120 (26 sentinel = 120) | **no — post-installation** | used only to build the Task B target |
| `avg_units_after` | int | Avg monthly electricity after, kWh | 0–311 | **no — post-installation** | used only to build the Task C target |
| `customer_given_rating` | int | Satisfaction 1–5 | 1–5 | **no — post-installation** | target D; forbidden elsewhere |
| `customer_referral` | bool | Customer referred the company | False 1,596 / True 904 | **no — post-installation** | target E; forbidden elsewhere |

## Derived targets (created in the cleaned dataset)
| Column | Definition |
|---|---|
| `temperature_drop` | `avg_roof_temperature_before − avg_roof_temperature_after` (°C) |
| `units_saved` | `avg_units_before − avg_units_after` (kWh/month; negative = use went up) |
| `high_rating` | `customer_given_rating ≥ 4` (threshold configurable) |

## Structural facts verified from the data
- No duplicate rows, no duplicate `building_id`, no near-duplicates ignoring the id.
- Roof type constrains method: no tile product on a sheet roof, no sheet product on a tile roof (0 of 2,500 records); concrete roofs accept all five.
- Cost is multiplicative: cost per area is driven by method × quality (basic ≈ 391–612, high ≈ 704–1,136 per area unit) with a smaller age/roof-type effect.
