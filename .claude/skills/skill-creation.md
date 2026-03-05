---
name: skill-creation
description: >-
  Create, design, or write .claude/skills/ files for a project. Use when
  scaffolding skills, adding a new skill, designing skill descriptions,
  writing trigger conditions, structuring skill workflows, or validating
  that a skill will trigger correctly. Also applies when reviewing or
  improving existing skills.
---

# Skill Creation Guide

Skills bridge architecture and agent behavior — they encode domain-specific
workflows so the coding agent knows HOW to perform tasks, not just WHAT the
domain is about.

**Key framing**: You are the *creator*. A different Claude Code instance is
the *consumer*. Every design choice — especially the description — must be
optimized for how that consumer agent encounters and triggers the skill, not
for your own understanding as the author.

---

## 1. Workflow-First Design

Skills must be **workflow-oriented**, not knowledge-oriented. If a section
doesn't tell the agent what to DO, it doesn't belong in a skill.

### Wrong: Knowledge dump
```markdown
# Paper Writing Skill

Academic papers have abstracts, introductions, methodology sections...
The IMRaD format is commonly used in scientific writing...
Citations should follow APA/Chicago/IEEE format...
```

### Right: Actionable workflow
```markdown
# Paper Writing Skill

## When to Use
- Writing a new section from scratch
- Revising a section based on feedback

## Steps
1. Read the paper outline from `docs/outline.md`
2. Identify the section's role in the paper's argument
3. Draft the section following the outline structure
4. Run citation density check: minimum 2 citations per paragraph
5. Verify all cited works exist in `references.bib`
6. Run quality gate: `python scripts/check_section.py <section>`

## Quality Gates
- [ ] Section follows outline structure
- [ ] Citation density >= 2 per paragraph
- [ ] No orphaned references
- [ ] Word count within target range
```

---

## 2. Skill Anatomy

### Progressive Disclosure — Three Levels

Skills load in stages. Design accordingly:

1. **Metadata (always loaded)** — YAML frontmatter (`name` + `description`).
   Claude sees every skill's description at all times to decide which to
   trigger. This is the most important part of the skill.
2. **Body (loaded on trigger)** — Markdown content below the frontmatter.
   Only loaded after the agent decides to use the skill.
3. **Bundled resources (on demand)** — Reference files, scripts, or assets
   the skill points to. Loaded when a step needs them.

### Two Formats

**Single-file** (most skills): `skill-name.md` — frontmatter + body in one file.

**Directory-based** (complex domains): A folder with `SKILL.md` as the entry
point, plus subdirectories for resources. Use when the skill needs reference
files too large to inline, or when the domain has multiple variants with
shared steps but different reference material.

```
.claude/skills/paper-writing/
├── SKILL.md              # Main skill (frontmatter + body)
├── references/           # Citation style guides, templates
├── scripts/              # Quality check scripts
└── assets/               # Example outlines, style samples
```

### Body Structure

1. **Frontmatter** — `name` and `description` (see Section 3 for description design)
2. **Trigger conditions** — When the agent should use this skill
3. **Prerequisites** — What must be true before the skill applies
4. **Numbered steps** — Concrete, ordered actions
5. **Quality gates** — Measurable checkpoints between major steps
6. **Exit criteria** — How the agent knows the workflow is complete

---

## 3. Consumer-Aware Description Design

The description is the **sole triggering mechanism**. Claude sees every
skill's description at all times and decides whether to load the full body.
If the description doesn't match the consumer's mental framing of their
task, the skill never triggers — no matter how good the body is.

### Creator vs Consumer

You (the creator) understand the skill deeply. The consumer only sees a
one-line description alongside dozens of others and must decide instantly
whether this skill is relevant.

The consumer encounters prompts from three sources:
- **End-user prompts**: "Add a skill for handling database migrations"
- **Orchestrator prompts**: L2 task titles like "Implement CSV parsing skill"
- **Agent self-prompts**: Internal reasoning like "I need to write tests for this module"

The description must match vocabulary the consumer would use in ALL these
contexts, not just the vocabulary you (the creator) would use.

### Write Pushy Descriptions

Claude under-triggers skills by default. Descriptions need to be "pushy" —
pack in synonyms, edge cases, and explicit contexts so the skill fires
whenever it should.

**Weak** (only matches exact phrasing):
```yaml
description: Guides creation of skill files.
```

**Pushy** (matches many phrasings the consumer might use):
```yaml
description: >-
  Create, design, or write .claude/skills/ files for a project. Use when
  scaffolding skills, adding a new skill, designing skill descriptions,
  writing trigger conditions, structuring skill workflows, or validating
  that a skill will trigger correctly. Also applies when reviewing or
  improving existing skills.
```

The pushy description covers: create, design, write, scaffold, add, review,
improve. It names sub-tasks (descriptions, triggers, workflows, validation)
and catches edge cases (reviewing, improving existing skills).

### Principles for Pushy Descriptions

- **Lead with verbs** — list every action this skill supports
- **Name sub-tasks** — not just "create skills" but what goes into it
  (descriptions, triggers, steps, quality gates)
- **Cover synonyms** — create/write/design/scaffold/author; include the
  ones a consumer might use
- **Include edge cases** — "Also applies when reviewing or improving" catches
  tasks the creator might not think of
- **Stay under 200 characters** — pack meaning, not filler

### Self-Check with Scenarios

Before finalizing a description, write 3-5 "would this trigger?" checks:

```
"Add a new skill for database migrations" → YES, "adding a new skill" covered.
"Review the paper-writing skill and fix the trigger" → YES, "reviewing" + "trigger conditions".
"Set up the project's .claude/ directory" → PARTIAL, broader than skills. Acceptable near-miss.
```

If any target prompt doesn't match, revise the description.

---

## 4. Lightweight Description Validation

After drafting the description, validate it systematically before finalizing.

### Step 1: Write Should-Trigger Queries (5-8)

Realistic prompts that SHOULD trigger the skill. Vary phrasing and source:

```
1. "Create a skill for handling API rate limiting"
2. "Add .claude/skills/ files for the paper-writing project"
3. "Write trigger conditions for the deployment skill"
4. "The migration skill isn't triggering — redesign its description"
5. "Scaffold skills for all workflow stages"
6. "Review existing skills and improve their quality gates"
```

### Step 2: Write Should-NOT-Trigger Queries (5-8)

Near-misses that should NOT trigger — related enough to test discrimination:

```
1. "Set up the .claude/settings.json file"
2. "Write the project's CLAUDE.md"
3. "Create a new CLI command in .claude/commands/"
4. "Design the project architecture"
5. "Write unit tests for the parser module"
```

### Step 3: Review and Revise

- Every should-trigger query should plausibly match the description
- No should-NOT-trigger query should strongly match
- Missing match → add vocabulary. Too-strong false match → narrow the scope

### Validation Checklist

- [ ] Description is under 200 characters
- [ ] Leads with action verbs (create, design, write, etc.)
- [ ] Covers at least 3 synonyms for the primary action
- [ ] Names specific sub-tasks, not just the top-level action
- [ ] Includes edge cases (review, improve, fix, validate)
- [ ] All should-trigger queries plausibly match
- [ ] No should-NOT-trigger query strongly matches
- [ ] Description reads naturally, not like a keyword list

---

## 5. Writing Style

### Use Imperative Form

Write instructions as commands: "Read the config file", not "The config file
should be read" or "You should read the config file."

### Explain the WHY

When a step isn't obvious, explain why it matters — but through brief context,
not heavy-handed MUSTs. "Check citation density (sparse citations weaken the
argument)" is better than "YOU MUST CHECK CITATION DENSITY."

### Define Output Formats with Examples

When a skill produces structured output, show a concrete example rather
than describing the format abstractly.

### Keep Lean

Every line should change agent behavior. Cut background info the agent
doesn't act on, redundant restatements, and long explanations where a
short example suffices.

### Generalize from Examples

Make examples illustrative of the pattern, not specific to one domain.
Note what pattern an example demonstrates so the agent can apply it elsewhere.

---

## 6. Anti-Patterns

### Missing trigger conditions
Without clear triggers, the agent either uses the skill for everything
or never uses it. State exactly when it applies.

### Missing quality gates
Without gates, the agent barrels through steps without checking quality.

### Knowledge without workflow
If a section doesn't tell the agent what to DO, cut it or rewrite it as
a step.

### Overly long skills
Skills over 500 lines are rarely read end-to-end. Split into multiple
skills or use directory-based format with reference files.

### No exit criteria
Without exit criteria, the agent doesn't know when to stop. Define what
"done" looks like with specific deliverables and verification.

### Writing for yourself instead of the consumer
If you write the description for your own understanding ("Guides creation
of workflow-oriented skill files"), you've optimized for the wrong audience.
Write for the agent scanning 20 descriptions and deciding which matches.

### Cramming trigger info into the body
The body loads AFTER the agent decides to trigger. Trigger scenarios,
edge cases, and synonyms in the body are useless — the agent already
committed before seeing the body. Put trigger info in the description.

---

## 7. Quick Reference

End-to-end checklist for creating a skill:

1. **Identify the workflow** — what sequence of actions does the agent need?
2. **Choose the format** — single-file `.md` or directory-based?
3. **Draft the body** — triggers, prerequisites, steps, gates, exit criteria
4. **Draft the description** — lead with verbs, cover synonyms, name sub-tasks
5. **Self-check with scenarios** — 3-5 "would this trigger?" prompts
6. **Validate systematically** — should-trigger and should-NOT-trigger query sets
7. **Review the body** — cut anything that doesn't change agent behavior
8. **Final check** — description under 200 chars, body under 500 lines, every
   section is actionable
