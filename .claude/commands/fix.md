---
description: Fix a blocked or failed task from the automation loop
---

## Automation Loop Error State

!`cat /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/.claude-loop-state 2>/dev/null || echo "No .claude-loop-state found. The loop may not have stopped with an error."`

## Recent Git History

!`git log --oneline -10 2>/dev/null || echo "No git history"`

## Your Job

The automation loop stopped and the human needs your help fixing the issue.

1. Read the error state above to understand what went wrong
2. Check what changed since the loop stopped: compare `last_agent_commit` from the state against HEAD
3. Investigate the relevant code, tests, and logs
4. Fix the underlying problem
5. Update meta files so the loop can resume cleanly:
   - tasks.json is at: /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/tasks.json
   - If the task status is "blocked" → set it to "in_progress" in tasks.json
   - If the task status is "in_progress" → leave it (the loop will re-verify)
   - Do NOT set status to "done" — the loop's verification agent handles that
6. Add an entry to progress.md (at /Users/hujunfeng/Documents/AMEX/claude-code-automation/projects/agentic_fraud_servicing/progress.md):
   ```
   ## HUMAN: [brief description of fix]
   [What was wrong and how it was resolved]
   ```
7. Commit all changes with a descriptive message. Do NOT include `Co-Authored-By` lines or any AI/Claude/Anthropic references in the commit
8. Tell the human to resume the loop

$ARGUMENTS
