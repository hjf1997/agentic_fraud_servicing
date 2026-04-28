# Scam Claim Types Strategy for LLM-Based Detection System

## Executive Summary

This document outlines the **SCAM-ONLY** claim type taxonomy designed for our LLM-based scam detection system. The classification focuses on **Authorized Push Payment (APP) Fraud** - transactions where the customer was psychologically manipulated into authorizing payment (scam), distinct from unauthorized transactions (fraud) and authorized but problematic transactions (dispute).

**Key Decision**: SCAM detection requires **multi-signal analysis** combining linguistic patterns, authorization evidence, payment characteristics, and behavioral indicators. Unlike FRAUD/DISPUTE, SCAM detection has **higher complexity** and **mandatory human review** due to liability sensitivity.

**Scope**: This system handles SCAM cases only. For FRAUD cases (unauthorized transactions), see [FRAUD_CLAIM_TYPES_STRATEGY.md](FRAUD_CLAIM_TYPES_STRATEGY.md). For DISPUTE cases (authorized but problematic), see [DISPUTE_CLAIM_TYPES_STRATEGY.md](DISPUTE_CLAIM_TYPES_STRATEGY.md).

**Scam Definition**: Transactions where the customer DID authorize payment but was psychologically manipulated or deceived by a scammer. Customer says "I was tricked into sending money" OR deceptively claims "I didn't authorize" when evidence shows they did.

---

## ⚠️ Critical Complexity Warning

**SCAM detection is fundamentally more complex than FRAUD or DISPUTE detection**:

| Complexity Factor | FRAUD/DISPUTE | SCAM |
|-------------------|---------------|------|
| **Detection Basis** | Technical evidence (transaction records, authentication logs) | Psychological analysis (social engineering, manipulation) |
| **Customer Statement Reliability** | Generally reliable | May be contradictory or deceptive |
| **Authorization Evidence** | Clear (yes/no) | Ambiguous (authorized but deceived) |
| **Liability Determination** | Straightforward (bank or merchant) | Complex (customer vs bank, regulatory gray area) |
| **Customer Sensitivity** | Medium (customer is victim) | **HIGH** (customer may feel blamed or embarrassed) |
| **False Positive Impact** | Process inefficiency | **Damages customer trust, potential legal risk** |
| **Human Review** | Optional for clear cases | **MANDATORY for all scam classifications** |

**Recommendation**: Use LLM for **scam indicator detection** and **risk scoring**, NOT for final scam determination. Always require human review before classifying as scam.

---

## The Three-Way Distinction

| Aspect | FRAUD | SCAM (APP Fraud) | DISPUTE |
|--------|-------|------------------|---------|
| **Customer Statement** | "I didn't make this charge" | "I was tricked into sending money" OR "I didn't authorize" (deceptive) | "I made this charge but..." |
| **Authorization** | Customer DID NOT authorize | Customer DID authorize (but was deceived) | Customer DID authorize |
| **Customer Intent** | No intent to pay | Intent to pay (based on deception) | Intent to pay (service issue) |
| **Fraudster** | Steals credentials/card | Manipulates customer psychologically | Legitimate merchant (service failure) |
| **Payment Method** | Card charge, unauthorized login | Wire transfer, P2P, crypto, gift card, check | Card charge (authorized) |
| **Liability** | Often bank (Reg E/Z) | Often customer (authorized it) | Often merchant (chargeback) |
| **Urgency** | HIGH (account security risk) | MEDIUM (money already sent) | MEDIUM (merchant issue) |
| **Customer Emotion** | Angry (security breach) | Embarrassed, ashamed, defensive | Frustrated (unmet expectation) |
| **Reason Codes** | Visa 10.x, MC 4837/4863 | Internal SCAM codes | Visa 13.x, MC 4853/4855 |

**Key Insight**: Customer may claim FRAUD ("I didn't authorize") to avoid liability, but evidence (wire transfer, 2FA passed, described payment process) reveals SCAM (authorized but deceived).

---

## Complete Scam Claim Types Mapping

### Tier 1: Scam Types (7 Categories - When Customer Admits Being Scammed)

These categories apply when the customer **acknowledges being deceived**.

#### Social Relationship Scams

| LLM Claim Type | Industry Term | Why Essential | Evidence Required |
|----------------|---------------|---------------|-------------------|
| **ROMANCE_SCAM** | Romance Fraud / Online Dating Scam | **15% of scams**. Cardmember says: "I met someone online who needed money for emergency." Ongoing emotional relationship, never met in person, repeated requests escalate over time. High emotional investment makes victims vulnerable. | Dating profile, communication history (messages, emails), payment timeline, recipient details (name, location, account), relationship duration, reason for each payment |
| **EMERGENCY_SCAM** | Emergency Fraud / Grandparent Scam / Family Emergency Scam | **5% of scams**. Cardmember says: "My family member called needing urgent money for bail/hospital/travel." Exploits family concern, creates time pressure, impersonates relative or authority. Urgency bypasses critical thinking. | Caller ID, phone number called, claimed emergency details, family member verification attempt, payment method demanded, time pressure tactics |

#### Financial Scams

| LLM Claim Type | Industry Term | Why Essential | Evidence Required |
|----------------|---------------|---------------|-------------------|
| **INVESTMENT_SCAM** | Investment Fraud / Ponzi Scheme / Crypto Scam | **25% of scams** (highest average loss $10k-$100k+). Cardmember says: "They promised guaranteed returns on crypto/forex investment." Fake trading platform, unrealistic returns (20%+ monthly), can't withdraw funds, pressure to invest more. | Platform name/URL, promised returns, account statements, communication with "broker", payment history, attempts to withdraw, referral source |
| **PURCHASE_SCAM** | Purchase Fraud / Advance Fee Fraud / Non-Delivery Scam | **10% of scams**. Cardmember says: "I paid for item online but never received it and seller disappeared." Payment outside secure platform (wire/P2P instead of card), new seller, too-good-to-be-true pricing. | Listing/advertisement, seller details, platform used, payment method, communication history, delivery promises, attempts to contact seller |
| **EMPLOYMENT_SCAM** | Employment Fraud / Fake Check Scam / Work-From-Home Scam | **5% of scams**. Cardmember says: "I got a job offer but had to pay fees/buy equipment first." Advance fees, fake check deposit with wire-back request, unrealistic salary for minimal work. | Job listing, employer details, fees requested, fake check image (if applicable), communication, job description, salary promise |

#### Authority Impersonation Scams

| LLM Claim Type | Industry Term | Why Essential | Evidence Required |
|----------------|---------------|---------------|-------------------|
| **IMPERSONATION_SCAM** | Impersonation Fraud / Vishing / Spoofing | **30% of scams** (highest volume). Cardmember says: "Someone from the bank/IRS/tech support called saying my account was compromised." Impersonates authority (bank, government, tech support), creates urgency, instructs transfer to "safe account". | Caller ID, phone number, claimed organization, urgency tactics, instructions given, authentication passed, transfer details, time of call |
| **INVOICE_SCAM** | Business Email Compromise (BEC) / Invoice Fraud / Vendor Impersonation | **10% of scams**. Cardmember says: "I received an invoice that looked legitimate but payment went to wrong account." Business context, altered payment details, fake vendor, email spoofing/compromise. Targets businesses. | Original invoice, altered invoice, email headers, vendor verification attempts, payment instructions, business relationship history, email compromise indicators |

**Tier 1 Scam Types Coverage**: 100% of cases where customer **acknowledges being scammed**.

---

### Tier 1: Meta-Detection (1 Category - Critical for Fraud vs Scam)

This category applies when customer **claims fraud but evidence suggests scam**.

| LLM Claim Type | Industry Term | Why Essential | Evidence Required |
|----------------|---------------|---------------|-------------------|
| **AUTHORIZATION_INCONSISTENCY** | Authorization Contradiction / First Party Fraud Indicator / Misrepresentation Detection | **15% of scam cases** (customers denying scam to avoid liability). Customer claims "I didn't authorize" BUT transcript/evidence reveals they DID authorize (described payment process, passed 2FA, wire/P2P payment). **CRITICAL for liability determination**. False classification damages customer trust. **MANDATORY HUMAN REVIEW**. | Transcript contradictions (fraud claim vs authorization description), payment method analysis (wire/P2P requires customer action), authentication evidence (2FA passed, known device/IP), timeline analysis (reported days/weeks later), customer knowledge level (knows recipient, payment details) |

**Critical Characteristics**:
- **NOT a scam type** - it's a detection pattern
- **Highest sensitivity** - false positive severely damages customer relationship
- **Liability impact** - determines if bank is liable (fraud) or customer is liable (scam)
- **Requires human judgment** - LLM flags, human decides
- **Empathetic approach** - customer may be embarrassed about being scammed

---

## Why SCAM Detection Is More Complex

### 1. Multi-Signal Analysis Required 🔍

FRAUD/DISPUTE can often be determined from **single signal**:
- FRAUD: "Customer didn't authorize" + No authentication passed = Clear fraud
- DISPUTE: "Customer authorized" + Merchant service issue = Clear dispute

SCAM requires **multiple signals converging**:

| Signal Category | What to Analyze | Complexity |
|-----------------|-----------------|------------|
| **Linguistic Patterns** | Social engineering tactics mentioned (urgency, secrecy, pressure, authority) | Requires NLP sentiment analysis |
| **Authorization Evidence** | 2FA passed, known device, customer-initiated payment, wire/P2P method | Contradicts fraud claim |
| **Payment Characteristics** | Wire/P2P/crypto (requires customer action), gift cards, multiple payments over time | Indicates customer involvement |
| **Timeline Analysis** | Reported days/weeks later (not immediately), ongoing payments over time | Suggests customer participated |
| **Customer Knowledge** | Knows recipient name/details, describes payment process, mentions communication | Indicates customer awareness |
| **Behavioral Indicators** | Embarrassment, defensiveness, blame-shifting, reluctance to admit deception | Requires emotional intelligence |

**LLM Challenge**: Must synthesize 6+ signal categories to determine scam likelihood. Any single signal can have alternative explanations.

---

### 2. Authorization Contradiction Detection 🎭

**The Core Challenge**: Customer claims FRAUD but evidence shows SCAM.

**Example Contradictions**:

| Customer Says (Fraud Claim) | But Transcript Also Reveals (Authorization) | Interpretation |
|-----------------------------|---------------------------------------------|----------------|
| "I didn't make this payment" | "They told me to transfer money to protect my account" | Customer DID initiate transfer (SCAM) |
| "I never authorized this" | "I entered the OTP code they asked for" | Customer passed 2FA (SCAM) |
| "Someone hacked my account" | "I sent a wire transfer to help my friend overseas" | Customer authorized wire (SCAM) |
| "I didn't recognize this charge" | "I was buying crypto on this investment platform" | Customer made purchase (SCAM) |
| "Card was stolen" | "I have my card right here, but I sent a Zelle payment" | Not card fraud, customer sent P2P (SCAM) |

**Detection Strategy**:

```python
# Pseudo-code for contradiction detection
def detect_authorization_contradiction(transcript, technical_evidence):
    """
    Returns: scam_likelihood (0.0-1.0), contradiction_details
    """

    # Signal 1: Customer fraud claim
    fraud_claim = extract_fraud_language(transcript)
    # "I didn't authorize", "I never made this", "Someone hacked"

    # Signal 2: Authorization indicators in transcript
    authorization_indicators = extract_authorization_language(transcript)
    # "They told me to transfer", "I entered the code", "I sent money"

    # Signal 3: Payment method requires customer action
    payment_method = technical_evidence.payment_method
    requires_customer_action = payment_method in ["wire", "p2p", "crypto", "gift_card", "check"]

    # Signal 4: Strong authentication passed
    auth_evidence = technical_evidence.authentication
    strong_auth_passed = auth_evidence.two_factor_passed and auth_evidence.known_device

    # Signal 5: Timeline inconsistency
    timeline = technical_evidence.timeline
    delayed_report = timeline.reported_days_after > 2

    # Signal 6: Customer knowledge level
    knowledge = extract_customer_knowledge(transcript)
    knows_details = knowledge.knows_recipient or knowledge.describes_payment_process

    # Calculate scam likelihood
    if fraud_claim and (
        authorization_indicators or
        (requires_customer_action and strong_auth_passed) or
        (delayed_report and knows_details)
    ):
        return high_scam_likelihood, contradiction_details

    return low_scam_likelihood, no_contradiction
```

**LLM Role**:
- ✅ Extract fraud language from transcript
- ✅ Extract authorization indicators from transcript
- ✅ Identify contradictions between the two
- ✅ Score scam likelihood (0.0-1.0)
- ❌ **DO NOT** make final scam determination (requires human review)

---

### 3. Liability Sensitivity & Customer Emotion 💔

**Why This Matters**:

| Scenario | Bank Action | Customer Reaction | Business Impact |
|----------|-------------|-------------------|-----------------|
| **False Positive (Call scam when it's fraud)** | Deny reimbursement | Angry, feels victimized twice, escalates to regulator | Customer churn, regulatory complaint, reputation damage |
| **False Negative (Miss scam, classify as fraud)** | Reimburse customer | Grateful (but bank absorbs loss) | Financial loss, precedent for future scam claims |

**Customer Emotions in Scam**:
- 😳 **Embarrassment**: "I can't believe I fell for this"
- 😠 **Defensiveness**: "It's not my fault, they tricked me"
- 😔 **Shame**: "I don't want to tell anyone"
- 😤 **Blame-shifting**: "The bank should have stopped this"

**Communication Challenges**:
- Customer may be **reluctant to admit** being scammed
- Customer may **deceptively claim fraud** to avoid liability
- Customer may **minimize their involvement** in payment
- Customer may be **emotionally vulnerable** and need support

**LLM Guidance**:
1. **Never accuse customer of lying** in generated responses
2. **Use empathetic language**: "It sounds like you may have been targeted by a scammer"
3. **Avoid victim-blaming**: Not "You should have known better" but "These scammers are sophisticated"
4. **Acknowledge emotions**: "I understand this is frustrating and concerning"
5. **Flag for human specialist**: Scam recovery team trained in empathetic engagement

---

### 4. Gray Area Cases - Ambiguous Authorization 🌫️

Some cases genuinely fall in gray area between FRAUD and SCAM:

| Scenario | Customer Action | Scammer Action | Classification Challenge |
|----------|----------------|----------------|-------------------------|
| **Coerced Authorization** | Passed 2FA under threat | "We've frozen your account, enter code NOW or lose money" | Did customer truly "authorize" under duress? |
| **Elderly/Vulnerable** | Easily confused, followed instructions | Exploits cognitive decline or vulnerability | Reduced capacity to consent? |
| **Sophisticated Spoofing** | Caller ID shows real bank number, website looks identical | Professional operation, hard to detect | Reasonable person standard? |
| **Partial Deception** | Authorized first payment (legitimate), later payments (scam) | Builds trust first, then pivots to scam | Which payments are scam? |
| **Business Context** | Authorized payment per "CEO email" | Business email compromise, spoofed email | Individual liability vs business process failure? |

**Regulatory Gray Area**:
- Reg E/Z protects against **unauthorized** transactions
- No clear protection for **authorized but deceived** transactions
- Some states have stronger consumer protection
- Reimbursement often decided case-by-case

**LLM Cannot Decide**:
- Whether coercion negates authorization
- Whether customer had capacity to consent
- Whether deception was sophisticated enough to excuse victim
- Whether bank has duty to reimburse

**Human Review Required**:
- Legal review for liability determination
- Customer service specialist for empathetic engagement
- Fraud analyst for pattern recognition
- Management approval for reimbursement decision

---

### 5. Social Engineering Tactic Detection 🎯

**What LLM Excels At**: Identifying psychological manipulation tactics in transcript.

**Common Tactics**:

| Tactic | How It Works | Transcript Indicators | Scam Type |
|--------|--------------|----------------------|-----------|
| **Urgency** | "You must act NOW or lose your money" | "They said I had to do it immediately", "I only had 10 minutes" | Impersonation, Emergency |
| **Secrecy** | "Don't tell anyone, not even the bank" | "They told me not to talk to anyone", "Keep this confidential" | Romance, Investment |
| **Authority** | "I'm calling from the fraud department" | "They said they were from the bank", "IRS agent" | Impersonation |
| **Fear** | "Your account will be closed/arrested" | "They said I'd be arrested", "Account would be frozen" | Impersonation, Invoice |
| **Greed** | "Guaranteed 20% monthly returns" | "Can't lose", "Risk-free", "Get rich quick" | Investment |
| **Emotional Connection** | Build trust over weeks/months | "We've been talking for months", "I trusted them" | Romance |
| **Complexity** | Technical jargon to confuse | "IP address compromised", "Blockchain security" | Tech support, Investment |
| **Social Proof** | "Everyone is doing this, don't miss out" | "My friend made $50k", "Thousands of investors" | Investment |

**LLM Detection Strategy**:

```python
# Example: Social engineering tactic extraction
TACTICS_DETECTED = {
    "urgency": [
        "immediately", "right now", "within 10 minutes", "before it's too late",
        "limited time", "act fast", "urgent", "time sensitive"
    ],
    "secrecy": [
        "don't tell anyone", "keep this confidential", "between us",
        "don't talk to the bank", "can't tell family", "secret"
    ],
    "authority": [
        "I'm calling from", "fraud department", "IRS agent", "official",
        "government", "law enforcement", "bank representative"
    ],
    "fear": [
        "arrested", "account will be closed", "lose all your money",
        "legal action", "frozen account", "criminal charges"
    ],
    "greed": [
        "guaranteed returns", "can't lose", "risk-free", "get rich",
        "double your money", "exclusive opportunity", "limited spots"
    ]
}

def extract_social_engineering_tactics(transcript):
    """
    Returns: List of tactics detected with confidence scores
    """
    tactics_found = []

    for tactic_name, keywords in TACTICS_DETECTED.items():
        matches = find_keywords_in_context(transcript, keywords)
        if matches:
            confidence = calculate_confidence(matches, transcript_length)
            tactics_found.append({
                "tactic": tactic_name,
                "confidence": confidence,
                "evidence": matches
            })

    return tactics_found
```

**Value**: High confidence social engineering tactics = High scam likelihood.

---

## LLM Capabilities & Limitations

### What LLM Can Do ✅

| Capability | Accuracy | Use Case |
|------------|----------|----------|
| **Linguistic pattern recognition** | High (85-90%) | Identify scam type based on customer description |
| **Contradiction detection** | High (85-90%) | Flag inconsistency between fraud claim and authorization indicators |
| **Social engineering tactic extraction** | High (80-90%) | Identify manipulation tactics mentioned in transcript |
| **Payment method analysis** | Very High (95%+) | Classify payment method (wire/P2P requires customer action) |
| **Timeline analysis** | High (85-90%) | Calculate delay between transaction and report |
| **Entity extraction** | High (85-90%) | Extract amounts, dates, recipient names, payment details |
| **Risk scoring** | Medium (70-80%) | Synthesize signals into scam likelihood score (0.0-1.0) |

### What LLM Cannot Do ❌

| Limitation | Why | Alternative |
|------------|-----|-------------|
| **Final scam determination** | Liability sensitivity, regulatory ambiguity, customer relationship risk | **Human review mandatory** |
| **Intent assessment** | Cannot truly know if customer was deceived vs complicit | Specialized fraud investigator |
| **Capacity evaluation** | Cannot assess elderly/vulnerable customer's ability to consent | Requires human judgment + customer history |
| **Empathetic engagement** | Can generate text but cannot adapt to customer's emotional state in real-time | Human customer service specialist |
| **Liability decision** | Legal and regulatory complexity, business policy decision | Legal review + management approval |
| **Reimbursement determination** | Business decision balancing customer satisfaction vs loss prevention | Management discretion |

---

## Multi-Signal Detection Framework

### Scam Likelihood Scoring Model

**Approach**: Combine multiple weak signals into strong confidence score.

```python
# Scam likelihood calculation framework
class ScamLikelihoodCalculator:
    """
    Calculates scam probability (0.0-1.0) based on multiple signals.
    """

    def calculate_scam_likelihood(self, transcript, technical_evidence):
        """
        Returns: scam_likelihood (0.0-1.0), signal_breakdown
        """

        signals = {}

        # Signal 1: Customer claims fraud (25% weight)
        signals['fraud_claim'] = self._detect_fraud_language(transcript)
        # 0.0 = No fraud claim, 1.0 = Strong fraud claim

        # Signal 2: Authorization indicators in transcript (30% weight)
        signals['authorization_indicators'] = self._detect_authorization_language(transcript)
        # 0.0 = No authorization mentioned, 1.0 = Clearly describes authorizing payment

        # Signal 3: Payment method requires customer action (20% weight)
        signals['payment_method'] = self._analyze_payment_method(technical_evidence)
        # 0.0 = Card charge (passive), 1.0 = Wire/P2P (active customer initiation)

        # Signal 4: Strong authentication passed (15% weight)
        signals['authentication'] = self._analyze_authentication(technical_evidence)
        # 0.0 = No auth / failed auth, 1.0 = 2FA passed + known device

        # Signal 5: Timeline inconsistency (10% weight)
        signals['timeline'] = self._analyze_timeline(technical_evidence)
        # 0.0 = Reported immediately, 1.0 = Reported >7 days later

        # Weighted sum
        scam_likelihood = (
            signals['fraud_claim'] * 0.25 +
            signals['authorization_indicators'] * 0.30 +
            signals['payment_method'] * 0.20 +
            signals['authentication'] * 0.15 +
            signals['timeline'] * 0.10
        )

        return scam_likelihood, signals
```

### Risk-Based Routing

| Scam Likelihood | Routing Decision | Rationale |
|-----------------|------------------|-----------|
| **0.0 - 0.3 (Low)** | Route as FRAUD | Likely genuine unauthorized transaction |
| **0.3 - 0.6 (Medium)** | Flag for specialist review | Ambiguous case, needs investigation |
| **0.6 - 0.8 (High)** | Route to scam recovery team | Likely scam, requires empathetic engagement |
| **0.8 - 1.0 (Very High)** | Flag for management + legal review | High liability sensitivity, reimbursement decision needed |

**Important**: Even Very High likelihood (0.8-1.0) still requires human confirmation before final classification.

---

## Implementation Recommendations

### Phase 1: Detection Infrastructure (Month 1-2)

**Goal**: Build multi-signal detection capability.

**Tasks**:
1. ✅ Implement linguistic pattern recognition (social engineering tactics)
2. ✅ Build contradiction detection (fraud claim vs authorization indicators)
3. ✅ Integrate technical evidence (payment method, authentication, timeline)
4. ✅ Develop scam likelihood scoring model
5. ✅ Create risk-based routing logic

**Success Criteria**:
- 80%+ accuracy in identifying social engineering tactics
- 85%+ accuracy in detecting authorization contradictions
- Scam likelihood score correlates with human judgment (0.75+ correlation)

---

### Phase 2: Human Review Workflow (Month 2-3)

**Goal**: Ensure all scam classifications reviewed by humans.

**Workflow**:

```
LLM Analysis → Scam Likelihood Score → Routing Decision
                                            ↓
                                      [Low 0.0-0.3]
                                            ↓
                                    Route as FRAUD (standard process)

                                      [Medium 0.3-0.6]
                                            ↓
                                    Flag for Specialist Review
                                            ↓
                        [Fraud Analyst investigates, determines FRAUD vs SCAM]

                                      [High 0.6-0.8]
                                            ↓
                                    Route to Scam Recovery Team
                                            ↓
                        [Empathetic engagement, explain scam, determine liability]

                                      [Very High 0.8-1.0]
                                            ↓
                            Flag for Management + Legal Review
                                            ↓
                        [Reimbursement decision, customer retention strategy]
```

**Key Roles**:
- **LLM System**: Detect patterns, score likelihood, route cases
- **Fraud Analyst**: Investigate ambiguous cases, gather evidence
- **Scam Recovery Specialist**: Empathetic customer engagement, scam education
- **Legal Team**: Liability determination, regulatory compliance
- **Management**: Reimbursement approval, policy decisions

---

### Phase 3: Customer Engagement Strategy (Month 3-4)

**Goal**: Empathetic scam victim support without victim-blaming.

**Communication Framework**:

```python
# Example: Empathetic scam response generation
SCAM_RESPONSE_TEMPLATES = {
    "acknowledge_scam": """
        I'm sorry to hear about this situation. Based on the details you've shared,
        it sounds like you may have been targeted by a sophisticated scam. These
        scammers are professionals who deceive people every day - this is not your fault.
    """,

    "explain_authorization": """
        From our records, I can see that you authorized this payment by [entering the
        security code / initiating the wire transfer]. Unfortunately, because you
        authorized the payment, even though you were deceived, the bank's ability to
        recover these funds is limited.
    """,

    "offer_support": """
        Here's what we can do to help:
        1. We'll file a fraud report with law enforcement
        2. We'll help you report this to the recipient bank (if wire transfer)
        3. We'll provide information on scam recovery resources
        4. We'll review your account for any additional unauthorized activity
    """,

    "avoid_victim_blaming": """
        ❌ DON'T SAY: "You should have known better" / "Why did you fall for this?"
        ✅ DO SAY: "These scams are very sophisticated" / "Many people are targeted"
    """
}
```

**Training Required**:
- Scam recovery specialists need training on:
  - Empathetic communication techniques
  - Common scam types and tactics
  - Liability boundaries (when bank can/cannot reimburse)
  - Scam reporting procedures (law enforcement, recipient bank)
  - Customer retention strategies

---

### Phase 4: Continuous Learning (Month 4+)

**Goal**: Improve detection accuracy over time.

**Feedback Loop**:

```
Human Review Decision → Update Training Data
                              ↓
                    [Case was actually FRAUD]
                              ↓
                    Update model to reduce false positives
                              ↓
                    [Case was actually SCAM]
                              ↓
                    Update model to improve scam detection
```

**Metrics to Track**:
- False positive rate (called scam, was fraud): **Target <5%**
- False negative rate (called fraud, was scam): **Target <15%** (more acceptable)
- Human review agreement rate: **Target >85%** (LLM score matches human determination)
- Customer satisfaction for scam cases: **Target >70%** (empathetic engagement)
- Reimbursement rate for scams: **Track trend** (business decision, not quality metric)

---

## Conclusion

**SCAM detection is fundamentally more complex** than FRAUD or DISPUTE detection due to:
1. ✅ **Multi-signal analysis required** (6+ signal categories vs 1-2 for fraud/dispute)
2. ✅ **Authorization contradiction detection** (customer says fraud, evidence shows scam)
3. ✅ **Liability sensitivity** (false positive damages customer trust, false negative costs money)
4. ✅ **Customer emotional state** (embarrassment, shame, defensiveness)
5. ✅ **Regulatory gray area** (no clear protection for authorized-but-deceived payments)

**LLM Role**:
- ✅ Detect linguistic patterns (social engineering tactics)
- ✅ Flag authorization contradictions
- ✅ Score scam likelihood (0.0-1.0)
- ✅ Route to appropriate team
- ❌ **NOT** final scam determination (human review mandatory)

**Key Recommendations**:
1. **Use LLM for detection, not determination** - Flag high-risk cases for human review
2. **Mandatory human review for all scam classifications** - Liability sensitivity too high for automation
3. **Empathetic customer engagement** - Scam victims need support, not blame
4. **Risk-based routing** - Different likelihood scores → different specialist teams
5. **Continuous learning** - Update model based on human review feedback

**Tier 1 SCAM Categories** (8 total):
- 7 Scam Types (when customer admits being scammed): ROMANCE_SCAM, INVESTMENT_SCAM, IMPERSONATION_SCAM, PURCHASE_SCAM, EMPLOYMENT_SCAM, INVOICE_SCAM, EMERGENCY_SCAM
- 1 Meta-Detection (when customer denies scam): AUTHORIZATION_INCONSISTENCY

**Coverage**: 100% of scam cases (both admitted and denied).

**Next Step**: Start with **AUTHORIZATION_INCONSISTENCY detection** (highest business impact - prevents false fraud reimbursements). Add specific scam types as volume justifies.

---

**Document Version**: 1.0
**Date**: 2026-03-03
**Author**: Fraud Investigation System Team
**Status**: Strategy Document for Phase 3 Implementation (after FRAUD and DISPUTE validated)
**Scope**: SCAM cases only (authorized but deceived transactions, may be disguised as fraud)
**Complexity Level**: ⚠️ **HIGH** - Requires multi-signal analysis + mandatory human review
