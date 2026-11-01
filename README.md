# Research question

How do the direction, magnitude, and pattern of prior financial outcomes affect an AI's subsequent risk-taking, and how does its behavior compare with experienced business decision-makers?

# Planned expansions

- Richer business context and more realistic financial mechanics
- More allocation options
- More years per history
- Additional history conditions
- Multiple model families and reasoning effort levels
- Independent repeated runs for each history
- Additional performance and risk metrics
- Comparison with experienced human business decision-makers

# Errors

- **429 — Rate limit:** Logs show frequent rejected attempts. Retries already wait between attempts. Find way to pace requests.
- **400 — JSON validation:** An intermittent `json_validate_failed` occurred despite `strict = True`; a subsequent run completed successfully. The cause is unknown.
- Implement error handling
