"""
reagent.experiments.ambiguity
=============================
Offline benchmark: can ReAgent's contract-policy distinguish committed
side effects from lost-response ambiguity better than naive retry?

Sub-modules
-----------
manifest   — pre-registered expectations (written before running)
simulator  — stateful service with authoritative commit store
faults     — deterministic fault injection modes
oracle     — independent oracle reading simulator store directly
policies   — 6 policy arm implementations
runner     — benchmark orchestration and metrics collection
"""
