# Fraud Case Opening Checklist

This document defines the criteria, verification requirements, and blocking rules
for opening a **fraud case** (unauthorized transaction claim) on an AMEX card account.

## Required Information

The following information **must** be collected before a fraud case can be opened:

- **Transaction details**: merchant name, transaction amount, transaction date
- **Date reported**: the date the cardholder first reported the unauthorized activity
- **Card status at time of transaction**: active, suspended, cancelled, or lost/stolen
- **Authentication method used**: chip+PIN, contactless, CNP (card-not-present), magnetic stripe, or unknown
- **Cardholder's account of events**: brief description of why the transaction is believed to be unauthorized

## Identity Verification

Before opening a fraud case, the CCP **must** verify the caller's identity:

- **Caller identity confirmed**: the caller has passed standard identity verification (security questions, one-time passcode, or callback verification)
- **Last 4 digits of card verified**: the caller has correctly stated the last 4 digits of the card number associated with the disputed transaction
- **If identity verification fails**: do NOT open a fraud case. Escalate to the Identity Verification team. A fraud case opened without verified identity may be voided.

## Eligibility Rules

A fraud case may only be opened if **all** of the following conditions are met:

- **Transaction age**: the disputed transaction must be within **120 days** of the transaction date
- **Card status**: the card must be **active** or **recently closed** (closed within the last 90 days) at the time of case opening
- **Transaction amount**: the disputed amount must exceed **$0.00** (zero-amount authorizations are handled by a different process)
- **Account standing**: the account must not be in collections or charged off

## Blocking Rules

A fraud case **cannot** be opened if any of the following conditions apply:

- **Previous merchant dispute**: the same transaction was previously opened as a merchant dispute case. The existing dispute must be **withdrawn or closed** before a fraud case can be opened on the same transaction.
- **Written confirmation**: the cardholder previously **confirmed the transaction in writing** (email, secure message, or signed statement). If the cardholder now claims it was unauthorized, escalate to the Special Investigations Unit (SIU) for review.
- **Duplicate case**: an active fraud case already exists for the same transaction. Check for existing cases before opening a new one.
- **Self-reported authorization**: the cardholder previously acknowledged authorizing the transaction during a recorded call. This requires SIU review before a new fraud case can be opened.
