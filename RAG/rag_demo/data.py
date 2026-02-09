"""Sample dataset, evaluation questions, and ground-truth answers for the RAG demo."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Source document
# ---------------------------------------------------------------------------
BRIEF: str = (
    "Big banks push back on proposal to raise FDIC coverage to $10 million for certain business accounts. "
    "A bipartisan proposal would raise federal deposit insurance coverage—currently capped at $250,000—up to $10 million "
    "for specific business transaction accounts, commonly described as noninterest-bearing transaction accounts used for operating "
    "purposes like payroll. What's being proposed: Reporting describes a Senate effort that would expand coverage far beyond the standard "
    "limit, but only for a targeted account category rather than all deposits. Who's involved: The bill is described as introduced by "
    "Sen. Bill Hagerty and Sen. Angela Alsobrooks. The proposal is also characterized as having prominent supporters in policy circles, "
    "including Sen. Elizabeth Warren, and mentions Scott Bessent in the broader push for changing deposit insurance. Why it's being "
    "discussed now: The proposal is framed as a response to deposit flight and bank-run dynamics highlighted by the 2023 failures of "
    "Silicon Valley Bank and Signature Bank, where uninsured depositors had strong incentives to move money quickly. Supporters' argument: "
    "Backers say many operating businesses routinely hold balances above $250,000 in transaction accounts to meet payroll and day-to-day "
    "obligations. Increasing coverage for that narrow account type, they argue, would reduce panic-driven withdrawals and help stabilize banks "
    "that rely on business operating deposits. Opponents' argument (notably large banks): Major banks and their representatives object to the "
    "proposal's cost and cross-subsidy effects, arguing that expanding coverage could increase liabilities for the deposit insurance system and "
    "shift costs unevenly. Critics also raise moral hazard concerns: higher coverage can weaken depositor incentives to monitor bank risk, potentially "
    "encouraging risk-taking or mispricing safety. Key bargaining detail described: Reporting notes that the Independent Community Bankers of America "
    "is described as endorsing the bill after securing an arrangement described as a 10-year cost-related exemption for its members. Disagreement on the "
    "number: The $10 million figure itself is presented as controversial, with discussion that some stakeholders view a lower cap (often described in the $1–$5 million range) "
    "as more reasonable if any increase is adopted."
)


def split_sentences(text: str) -> list[str]:
    """Split *text* on sentence-ending punctuation."""
    return re.split(r"(?<=[.!?])\s+", text.strip())


DATA: list[str] = [s.strip() for s in split_sentences(BRIEF) if s.strip()]

# ---------------------------------------------------------------------------
# Evaluation questions & expected answers
# ---------------------------------------------------------------------------
QUESTIONS: list[str] = [
    # Who / what / when
    "Who introduced the Senate bill described as raising deposit insurance to $10 million for certain accounts?",
    "What is the proposed new insurance cap amount and what is the current standard cap referenced in coverage?",
    "What category of accounts is the proposal focused on (as described in sources)?",
    "Which two 2023 bank failures are cited as motivating context for the proposal?",
    "Which trade group is described as endorsing the bill after securing a cost-related exemption, and how long is that exemption described to last?",
    "Name two prominent public figures reported as pushing for higher deposit insurance caps in this context.",
    # Mechanism and policy design
    "According to industry commentary, what rollout structure is described for the $10 million concept?",
    "What is the main 'level playing field' argument supporters make (as reported)?",
    "What is one core cost objection from large banks mentioned in reporting?",
    "What 'moral hazard / distributional' critique do some opponents raise about raising the cap to $10 million?",
    "What alternative cap range is mentioned by some stakeholders instead of $10 million?",
    # Numeric detail
    "What was the last major increase to the standard FDIC insurance cap referenced in reporting, and what did it change from/to?",
]

GROUND_TRUTH: list[str] = [
    "Bill Hagerty and Angela Alsobrooks",
    "$10 million (for certain accounts); $250,000 per depositor per bank",
    "Noninterest-bearing transaction accounts / specific business transaction accounts (often framed as payroll-type operating accounts)",
    "Silicon Valley Bank and Signature Bank",
    "Independent Community Bankers of America; 10 years",
    "Scott Bessent and Elizabeth Warren",
    "A phased-in approach (described as 10 years) and exclusions for the largest banks (e.g., U.S. GSIBs)",
    "Expanded coverage could reduce incentives for depositors to flee to very large banks during stress, potentially stabilizing small/midsize banks.",
    "Large banks argue they'd bear much of the cost (as major contributors to the insurance fund) while disputing that the change improves safety.",
    "Critics argue it would benefit a small slice of accounts and could shift risk to others / create incentives that increase systemic exposure.",
    "$1 million to $5 million",
    "In 2008, from $100,000 to $250,000",
]
