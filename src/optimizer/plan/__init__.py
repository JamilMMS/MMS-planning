"""Plan optimizer (METHODOLOGY section 4): candidate set, constraints, S3 MILP, S1/S2/S4 reach
heuristics (lazy greedy + local search), constraint verification, KPIs, scenarios and frontier.

Entry point: ``python -m optimizer.plan.run --config config/plan_config.yaml``.
"""
from .constraints import verify_constraints  # noqa: F401
from .problem import Problem, build_candidates, build_problem  # noqa: F401
