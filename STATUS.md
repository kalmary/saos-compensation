# Project status

## Summary

Run the acquisition pipeline through the get-data command. The acquisition components are introduced in dependency order; full pipeline integration remains pending.

## Log

- 2026-11-03: Normalize judgment text and parse PLN amounts; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Write atomic records within a cumulative storage limit; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Validate and paginate official SAOS dump responses; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Resolve court localities from official population data; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Encode text with cached models and compare embedding vectors; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Match award sentences against target and payment patterns; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Validate extraction settings and cache model decisions; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Resolve awarded principal and earlier payments separately; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Validate admission and export separate feature and award records; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Extract source-backed factual features with leakage checks; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Resume judgment downloads only after durable record output; validated with its own locked environment and the offline tests available at this layer.
- 2026-11-03: Run the acquisition pipeline through the get-data command; validated with its own locked environment and the offline tests available at this layer.
