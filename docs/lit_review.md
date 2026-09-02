# Literature & Prior Art Notes

## Existing commercial systems (know these — judges may too)
- **AutoGrid Flex™ (now Uplight)** — deployed on Tata Power Delhi Distribution
  (TPDDL), covering 7M+ people. Behavioral demand response: notifies customers,
  verifies participation, rewards financially. Grew from 2,000 → 100,000 customers
  over 3 years, ~1MW → projected 100MW load shed. Ranked #1 DERMS platform globally
  (Guidehouse Research).
- **Known gap even AutoGrid admits**: strong at predict/optimize/control at scale,
  but enrolling/connecting millions of small/informal consumer devices is the hard
  part. → Our angle: lightweight layer for consumers enterprise DERMS doesn't
  economically reach.
- **PTC India + AutoGrid MOU** — explicitly frames India's demand as getting
  "peakier" due to rising HVAC/AC load — same language as our problem statement.

## Key academic gap (cite this — it's your strongest novelty justification)
Multiple 2024-2025 papers on AI-enhanced VPP optimization note that most AI-for-grid
work treats **load forecasting, dispatch, and demand response as loosely coupled
components** — forecasting errors aren't corrected by scheduling, and DR isn't
dynamically aligned with real-time constraints. This directly justifies building a
forecast→optimize→adapt **closed loop**, which most hackathon teams won't bother with
(they'll build a forecaster and stop).

## Forecasting gap
Most reviewed load-forecasting literature focuses on **point accuracy**, not
uncertainty-aware/probabilistic forecasting for peak demand. → justifies our P10/P50/P90
quantile regression approach as a genuine differentiator, not just decoration.

## RL/DR review papers (cite 2-3 of these in your submission)
- "Reinforcement learning for demand response: A review of algorithms and modeling
  techniques" (ScienceDirect) — notes most RL-DR work is single-agent, stationary-
  environment — real grids are neither.
- "Applications of Reinforcement Learning in Deregulated Power Market: A
  Comprehensive Review" (arXiv 2205.08369)
- "Distributed Energy Management and Demand Response in Smart Grids: A Multi-Agent
  Deep RL Framework" (arXiv 2211.15858) — DQN-based, joint DR + distributed energy
  management for prosumers.
- "Fair Allocation Based Soft Load Shedding" (arXiv 2002.00451) — directly relevant
  to our fairness-floor constraint design.

## Patents (landscape awareness, not reinvention)
- US 8,392,031 — "System and method for load forecasting" (EMS/DMS integration,
  advanced metering + demand response).
- Enel X / Tesla — patents on AI platforms setting dynamic electricity rates using
  market data, weather, and consumption patterns.

## One-line pitch framing
"Enterprise DERMS platforms like AutoGrid/Uplight already prove DR works at scale in
Delhi, but the literature shows a persistent integration gap between forecasting,
dispatch, and real-time DR — and industry solutions still underserve small/informal
consumers. We build a lightweight forecast-optimize-adapt loop with uncertainty-aware
forecasting and a hard fairness constraint, closing both the technical gap (loose
coupling) and the adoption gap (low-cost, small-consumer accessible)."
