# Agent Instructions

You operate within the **WAE Framework (Workflows, Agents, Engine)**.

This architecture separates reasoning from execution:
- Probabilistic AI handles coordination.
- Deterministic code handles computation.

This separation ensures reliability, reproducibility, and production-grade discipline.

If each step is 90% accurate, five steps compound to 59%. Offload execution to deterministic modules. Focus on orchestration.

---

# Architecture

## 1. Workflows (Intent)

Location: `workflows/`

Markdown SOPs defining:
- Objective
- Required inputs
- Expected outputs
- Engine modules to call
- Edge cases
- Validation requirements
- Performance or cost constraints

Workflows define **what must happen**, not how.

Do not overwrite or create workflows unless instructed.

---

## 2. Agents (Coordination)

Location: `agents/`

Agents:
- Read workflows
- Determine execution order
- Call engine modules
- Handle failures
- Ask clarifying questions when needed

Agents must NOT:
- Reimplement deterministic logic
- Manually compute metrics that exist in engine
- Bypass validation
- Embed business logic outside engine

Agents orchestrate. Engine executes.

---

## 3. Engine (Deterministic Execution)

Location: `engine/`

All computational logic lives here.

Everything inside must be:
- Deterministic
- Testable
- Reproducible
- Version-controlled


Secrets live only in `.env`. Never hardcode credentials.

---

# Operating Rules

## 1. Reuse Before Building

Check `engine/` before creating new modules.
Avoid duplicated logic.

---

## 2. Determinism Over Guesswork

If logic is repeatable, it belongs in `engine/`.

Agents must not:
- Approximate transformations
- Simulate optimization
- Recalculate metrics manually

Call the correct module.

---

## 3. Reproducibility Is Mandatory

All modeling must:
- Use explicit random seeds
- Use defined CV folds
- Maintain feature ordering
- Avoid implicit randomness

No silent variability.

---

## 4. Validate Before Accepting Output

Always validate:
- Schema and shape
- Uniqueness constraints
- Metric consistency
- Ordinality (if applicable)

Fail fast. Silent corruption is unacceptable.

---

# Failure Protocol

When something breaks:

1. Read the full error trace.
2. Identify the failing layer:
   - Workflow
   - Agent
   - Engine
3. Fix at the correct layer.
4. Retest deterministically.
5. Update workflow if institutional knowledge changed.

Never patch around engine failures in the agent layer.

If paid APIs are involved, confirm before rerunning.

---

# Self-Improvement Loop

Every failure must strengthen the system:

1. Identify root cause
2. Fix engine logic
3. Verify deterministically
4. Update workflow
5. Move forward

The system compounds reliability over time.

---

# Output Discipline

## outputs/
Production-ready deliverables only.

## outtest/
Experiments, comparisons, validation runs.

## .tmp/
Fully regenerable. Nothing here is durable.

---

# Production Standards

Engine modules must:
- Avoid global state
- Avoid side effects unless intentional
- Be unit-testable
- Log meaningful operations
- Use clean function signatures

Agents must:
- Call engine modules, not replicate them
- Respect validation boundaries
- Maintain clean separation of concerns

---

# Bottom Line

Workflows define intent.  
Agents coordinate intelligently.  
Engine executes deterministically.  

Validate everything.  
Keep outputs clean.  
Keep intermediates disposable.  
Build like this system will scale.


