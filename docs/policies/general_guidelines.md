# General Case Opening Guidelines

This document defines cross-cutting rules, priority handling, conflict resolution,
documentation requirements, and escalation triggers that apply to **all case types**
(fraud, dispute, and scam).

## Priority Rules

Certain situations require the CCP to prioritize specific actions:

- **Lost or stolen card**: if the cardholder reports their card as lost or stolen, **immediately initiate card replacement** and prioritize opening a fraud case for any unauthorized transactions. Do not delay card replacement to gather dispute information.
- **Chip+PIN transaction flagged as fraud**: if the disputed transaction was authenticated via **chip+PIN**, the CCP **must inform the cardholder** of this fact before opening a fraud case. Chip+PIN transactions have a higher burden of proof. Document the cardholder's response to this information.
- **Multiple transactions**: if the cardholder is reporting multiple unauthorized transactions, open a **single fraud case** covering all related transactions rather than individual cases per transaction.
- **Active card compromise**: if there is evidence of ongoing unauthorized activity (multiple recent transactions the cardholder does not recognize), escalate to the **Fraud Operations team** for immediate card block and expedited investigation.

## Case Type Conflicts

A single transaction can only have **one active case** at a time:

- **One case per transaction**: a transaction cannot simultaneously be under both a fraud case and a merchant dispute. If the cardholder wants to change the case type, the existing case must be **closed or withdrawn** before a new case of a different type can be opened.
- **Type change procedure**: if a cardholder initially filed a merchant dispute but now believes the transaction was fraudulent (or vice versa), the CCP should: (1) document the reason for the type change, (2) close the existing case with reason "type change requested", (3) open the new case type with a cross-reference to the original case.
- **Scam cases**: scam cases follow the fraud case checklist with additional documentation of the social engineering method. A scam case and a fraud case are considered the same case type for conflict purposes.

## Documentation Requirements

All case openings must meet these documentation standards:

- **Verbal confirmation**: the cardholder must **verbally confirm** their intent to open a case during the recorded call. The CCP must note in the transcript that confirmation was obtained.
- **Summary of claim**: the CCP must record a brief summary of the cardholder's claim in the case notes, including: what happened, when it happened, and what resolution the cardholder expects.
- **Evidence collection**: if the cardholder has supporting documents (receipts, emails, screenshots), note what is available and instruct the cardholder to upload them through the secure portal within **10 business days**.
- **Disclosure**: the CCP must inform the cardholder that: (1) the case will be investigated, (2) a provisional credit may be issued during investigation, and (3) if the investigation finds the claim is not valid, any provisional credit will be reversed.

## Escalation Triggers

The following situations require **immediate escalation** beyond standard case opening:

- **High-value transactions**: cases involving transactions over **$10,000** require **supervisor review** before the case can be finalized. The CCP may open the case but must flag it for supervisor approval.
- **Elderly or vulnerable cardholders**: if the cardholder is identified as elderly (65+) or vulnerable (mentions confusion, cognitive difficulty, or being pressured by a third party), route to the **Special Handling team** for enhanced support.
- **Suspected first-party fraud**: if the CCP observes indicators of potential first-party fraud (inconsistent story, evidence contradicts claims, cardholder becomes evasive when presented with facts), do NOT confront the cardholder. Document observations in case notes and flag for **SIU review**.
- **Repeat claimants**: if the cardholder has **3 or more** fraud or dispute cases in the past 12 months, flag the case for **pattern review** by the Risk Analytics team.
- **Law enforcement involvement**: if the cardholder mentions a police report or law enforcement investigation, note the report number and jurisdiction in the case file and flag for **Legal team** coordination.
