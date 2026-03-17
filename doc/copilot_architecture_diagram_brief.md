# AMEX Realtime Fraud Copilot — Architecture Diagram Brief

Use this brief to generate a clean, professional architecture diagram for the
Realtime Copilot. The audience is non-technical leadership. Use simple labels,
clear arrows, and a top-to-bottom flow. Avoid technical jargon. Use AMEX blue
(#006FCF) as the primary color.

---

## What the Copilot Does

When a cardmember calls AMEX to dispute a charge, a human agent (Contact Center
Professional, or CCP) handles the call. Our AI Copilot listens to the conversation
in real time and helps the CCP make better decisions, faster.

---

## Diagram Layout (top to bottom)

### Top: The Two People on the Call

Show the Cardmember (customer) on the left and the CCP (AMEX agent) on the right,
connected by a line labeled "Live Phone Call". The conversation flows down into the
AI Copilot.

### Middle: The AI Copilot (hub-and-spoke — the visual centerpiece)

The Copilot has a central brain (Orchestrator) that coordinates four specialist
AI agents. Show this as a hub-and-spoke pattern — the Orchestrator in the center,
with four agents arranged around it in a circle or semicircle. Each agent has a
short name and a one-line plain-English description:

1. **Claim Extractor** — "What is the customer claiming?"
   (e.g., "I didn't make this purchase", "I never signed up for this subscription")

2. **Question Advisor** — "What should the agent ask next?"

3. **Data Retriever** — "What does the account history show?"
   Connects down to three small data icons:
   - Transaction History (past purchases)
   - Login & Auth Records (how the card was used)
   - Customer Profile (account details)

4. **Hypothesis Scorer** — "What most likely happened?"
   Scores four possibilities shown as a small bar chart or badges:
   - Unauthorized Fraud (someone else used the card)
   - Scam (customer was tricked by a fraudster)
   - Friendly Fraud (customer is misrepresenting what happened)
   - Merchant Dispute (legitimate complaint about the merchant)

Arrows go from the Orchestrator outward to each agent and back, showing the
Orchestrator asking each agent and collecting their answers.

### Bottom: Output to CCP

An arrow goes from the Copilot back up to the CCP, delivering a panel labeled
**"Copilot Suggestion"** with these items:

- Suggested next questions (1-3 questions)
- Risk flags (warnings for the agent)
- Hypothesis scores (the four-bar chart)
- Safety guidance

This panel is what the CCP sees on their screen during the call.

---

## Step-by-Step Flow (for arrow labels or a numbered sequence)

For every turn of the conversation:

1. Customer or CCP speaks
2. **Orchestrator** receives the text and activates agents:
   - **Claim Extractor** identifies any new claims from the customer
   - **Data Retriever** looks up account history (transactions, logins, profile)
   - **Question Advisor** suggests the best questions to ask next
   - **Hypothesis Scorer** weighs all evidence and scores the four explanations
3. Orchestrator assembles the **Copilot Suggestion** and delivers it to the CCP's screen

---

## Visual Style Notes

- AMEX blue (#006FCF) for the Copilot hub and primary elements
- Dark navy (#00175A) for headers and labels
- Light gray (#F7F8FA) for background sections
- White cards with subtle shadows for each agent
- The hub-and-spoke pattern is the visual centerpiece — make it prominent
- Keep it to ~12-15 elements total — clean and scannable in 10 seconds
- The four hypothesis scores as a horizontal bar chart is a strong visual anchor
- The CCP and Cardmember at the top humanize the diagram — show people icons

---

## Key Message

"Our AI Copilot assists AMEX agents in real time during dispute calls. For every
turn of the conversation, it extracts claims, looks up account data, suggests questions,
and scores four possible explanations — helping agents
make faster, more accurate decisions."
