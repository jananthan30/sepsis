# Track Registry

This file tracks all development tracks managed by Claude Maestro.

## Active Tracks

### [001-jepa-world-model] JEPA World Model for Sepsis Prediction
- **Status:** active
- **Type:** feature
- **Created:** 2025-12-26
- **Branch:** `feature/jepa-world-model`
- **Spec:** [spec.md](tracks/001-jepa-world-model/spec.md)
- **Plan:** [plan.md](tracks/001-jepa-world-model/plan.md)
- **Summary:** Implement PyTorch-based JEPA architecture for self-supervised learning of patient physiology, with A/B comparison against existing BiLSTM

## Completed Tracks

(No completed tracks yet)

## Archived Tracks

(No archived tracks yet)

## Statistics

| Metric | Count |
|--------|-------|
| Total Tracks | 1 |
| Completed | 0 |
| In Progress | 1 |
| Blocked | 0 |
| Archived | 0 |

---

## Track Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRACK LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   /maestro:new "feature"                                        │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐                                                 │
│   │  CREATED  │  → spec.md drafted                              │
│   └─────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐                                                 │
│   │  PLANNED  │  → plan.md with phases/tasks                    │
│   └─────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐                                                 │
│   │   ACTIVE  │  → /maestro:implement working                   │
│   └─────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐                                                 │
│   │ COMPLETED │  → All tasks done, verified                     │
│   └─────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐                                                 │
│   │ ARCHIVED  │  → /maestro:archive (optional)                  │
│   └───────────┘                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Commands Quick Reference

| Command | Description |
|---------|-------------|
| `/maestro:new "description"` | Create a new development track |
| `/maestro:list` | List all tracks with status |
| `/maestro:status` | Show current progress of active work |
| `/maestro:implement` | Execute tasks from current track |
| `/maestro:task <id>` | Start working on a specific task |
| `/maestro:sync` | Synchronize context with codebase changes |
| `/maestro:revert` | Revert a track, phase, or task |
| `/maestro:archive` | Archive completed tracks |
| `/maestro:restore` | Restore archived tracks |
