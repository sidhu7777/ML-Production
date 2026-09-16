# Prediction

This is a legacy prediction API wrapper registered under `/api/prediction`. The current LTE production paths are `tools/lte_prediction`, `tools/lte_prediction_offset`, and `tools/lte_prediction_optimised`; use this module only where the existing frontend/API contract still calls it.

## Public API

Base prefix:

```text
/api/prediction
```

Check `routes.py` for the exact legacy request/response contract before extending this module.

## Code Map

| File | Role |
| --- | --- |
| `routes.py` | Flask API routes. |
| `services.py` | Legacy prediction service implementation. |
