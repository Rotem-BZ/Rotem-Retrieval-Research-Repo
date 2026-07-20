---
name: create-experiment-card
description: Create or update a very small Markdown experiment card containing only a basic description of the experiment and a hypothesis supplied explicitly by the user. Use when documenting a proposed retrieval experiment or reducing an existing experiment card to its essential idea and hypothesis.
---

# Create Experiment Card

Write a short `experiment.md` that records only what the experiment is and the user's hypothesis.

## Required input

Obtain the hypothesis directly from the user before writing the card.

- If the user has not stated a hypothesis, ask them for it and stop. Do not infer, propose, complete, or invent one from the experiment description, repository, configs, prior cards, or general knowledge.
- If the user's wording is ambiguous, ask them to clarify it. Do not silently make it more specific or directional.
- Preserve the meaning of the user's hypothesis. Only make superficial wording corrections when they do not change the claim.

## Workflow

1. Read [references/card-template.md](references/card-template.md).
2. Confirm that the user supplied both an experiment description and a hypothesis. Ask only for missing information.
3. Inspect repository files only as needed to identify the project, experiment name, and concrete comparison accurately.
4. Create or update `projects/<project>/experiments/<experiment-slug>/experiment.md` using the template.
5. Keep the finished card brief. Do not add execution plans, commands, configs, metrics, decision rules, artifact lists, validity sections, analysis plans, results, owners, dates, or other preregistration material.

## Completion criteria

The card contains a compact description of what will be tested and the hypothesis explicitly supplied by the user, with no agent-authored hypothesis or additional planning sections.
