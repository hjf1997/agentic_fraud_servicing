---
description: Steer project direction mid-project (skip tasks, add milestones, edit descriptions)
---

## Current Task Tree

!`cat /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/tasks.json`

## Recent Progress

!`tail -40 /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/progress.md`

## Recent Git History

!`git log --oneline -10 2>/dev/null`

## Your Job

The human wants to steer the project direction. They will describe what they want
to change. Choose the appropriate operation(s) from the list below.

Task files are at: /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing

### Available Operations

**a) Add HUMAN directive to progress.md**
Append a `## HUMAN: Steering` entry to progress.md (at /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/progress.md)
with the human's instructions. This is the lightest intervention — no structural
changes, just guidance for future sessions.

**b) Skip or unskip tasks**
Set a task's status to `"skipped"` (or back to `"pending"` to unskip) in
/Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/tasks.json.
- Never skip a task with status `"done"` or `"in_progress"`
- If skipping an L1 that has L2 subtasks, skip ALL its L2 subtasks too
- If unskipping an L1, unskip ALL its L2 subtasks too

**c) Add new L1 milestones**
Insert new L1 tasks into /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/tasks.json. New L1s must follow this schema:
```json
{
  "id": <next_integer>,
  "level": 1,
  "parent_id": null,
  "title": "Short descriptive title",
  "description": "Detailed description with enough context for decomposition...",
  "acceptance_criteria": ["Verifiable criterion"],
  "verification_steps": [],
  "status": "pending",
  "notes": ""
}
```
Choose an ID that maintains ordering (use the next available integer, or
renumber pending tasks if insertion order matters).

**d) Modify pending task descriptions**
Edit the `title`, `description`, `acceptance_criteria`, or `verification_steps`
of a pending task in /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/tasks.json.
- Never modify tasks with status `"done"`
- If the task has existing L2 subtasks, WARN the human: "This L1 has L2
  subtasks that were decomposed based on the original description. Changing
  the L1 description may make existing L2s inconsistent. Consider also
  reviewing or removing the L2s."

**e) Reorder L1 priorities**
Renumber pending L1 task IDs to change execution order.
- Never renumber done or in_progress tasks
- Update all `parent_id` references in L2 tasks to match new L1 IDs
- Update all L2 IDs to match their new parent prefix (e.g., if L1 3 becomes
  L1 5, then L2 3.1 becomes L2 5.1)

**f) Update CLAUDE.md conventions**
Modify the project's CLAUDE.md to change coding conventions, build commands,
or other project-level settings.
- Do NOT modify the session type instructions or task protocol sections
- Only modify: Code Conventions, Build and Development Commands, or add
  new sections below the existing ones

**g) Update ARCHITECTURE.md**
Modify the project's ARCHITECTURE.md (at /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/ARCHITECTURE.md) to
reflect changed architectural decisions, revised tech stack choices, updated
directory structure, or new component interactions.
- Update any section: vision, architecture overview, directory structure,
  tech stack, design decisions, data flow, data models, non-functional requirements
- Common scenarios: changed tech choices, revised directory layout, added or
  removed system components, updated data flow, new constraints discovered

### Rules

1. **Document everything**: After making changes, add a `## HUMAN: Steering` entry
   to progress.md (at /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/progress.md) describing what was changed and why.

2. **Commit format**: Commit all changes with:
   ```
   steer(<scope>): <what changed>

   <why the human requested this change>

   CHANGE: <summary>
   ```
   Do NOT include `Co-Authored-By` lines or any AI/Claude/Anthropic references.

3. **Never modify done tasks**: Tasks with status `"done"` are immutable.

4. **Never modify harness files**: Do NOT touch `.claude/settings.json`.

5. **Validate after changes**: After modifying tasks.json, verify it is valid JSON
   and that all L2 `parent_id` references point to existing L1 tasks.

6. **Tell the human what changed**: After committing, summarize exactly what was
   modified so the human can verify before restarting the loop.

$ARGUMENTS
