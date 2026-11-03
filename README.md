# Research question

How do the direction, magnitude, and pattern of prior financial outcomes affect an AI's subsequent risk-taking, and how does its behavior compare with experienced business decision-makers?

# Proposed experimental scope

- **3 models:** one API-accessible model from each of three leading providers, selected based on documented use or recommendation in financial applications. Exact models remain to be confirmed. The experiment tests the underlying models, not the providers' complete financial-services products.
- **Maximum reasoning effort:** use each model's highest publicly supported reasoning-effort setting and hold it constant across histories. This standardizes the selection rule, not the amount of computation across models.
- **9 financial histories:** one neutral baseline plus gains/losses × small/large amounts × sudden/streak patterns.
- **5 independent repetitions per history:** each session starts with fresh cash and no previous decisions. Review observed variability before deciding whether more repetitions are needed.
- **5 annual allocation decisions per session.**

**3 models × 9 histories × 5 repetitions × 5 annual decisions = 675 decisions.**

Keep starting cash, investment options, instructions, and business context consistent across models and history conditions. Use the same realized fund returns for corresponding years across histories and repetitions to isolate variation in model decisions.

The first annual allocation is the primary outcome: starting cash and investment options are identical, so prior financial history is the experimental difference within each model. Later years explore whether differences persist as allocations and realized results accumulate. Measure risk-taking through allocations and the known risk characteristics of the investment options; use model explanations as supporting evidence.

Conclusions apply to the selected models and configurations, not entire model families. Comparison with experienced business decision-makers is a separate phase using the same decision task; participant eligibility and recruitment remain to be defined.

## Development needed for the full experiment

- Develop richer business context, more realistic financial mechanics, and additional allocation options. Finalize these before data collection and keep them consistent across conditions.
- Integrate the selected providers and record each model's exact configuration and usage.
- Implement allocation validation, error handling, request pacing, and resume support before larger experiments.
- Define the performance and risk metrics used to evaluate decisions.
- Measure usage before larger runs to estimate API costs.

# Current implementation

The current implementation is a limited development version using GPT-OSS 20B through Groq at medium reasoning effort. It includes 9 financial histories, 3 repetitions, and 5 annual decisions per session (135 decisions), rather than the full experiment's selected models, maximum effort, and five repetitions.

Business context is deliberately minimal: the model manages a company's cash using a supplied financial history and fixed investment rules. Other business financial needs are treated as covered; revenue, debt, staffing costs, and other operating constraints are not modeled. Allocation choices are limited to three funds: a lower-risk investment, a higher-risk investment, and a zero-return cash option. This version supports development and testing before the full experiment is finalized.

# Possible future extensions

- More years per history
- Additional history conditions
- Additional models within each family and comparisons across reasoning effort levels
- Different investment outcome sequences (seeds)

# Errors

- **429 — Rate limit:** Logs show frequent rejected attempts. Retries already wait between attempts. Find way to pace requests.
- **400 — JSON validation:** An intermittent `json_validate_failed` occurred despite `strict = True`; an identical request can succeed on another attempt. Malformed output was reproduced, but the provider's internal cause is unknown.
- **400 — Output truncation:** A separate error reported that the output limit was reached before the required JSON was complete. Requests now explicitly set `max_output_tokens = 10000`; this has not established that truncation is resolved.
