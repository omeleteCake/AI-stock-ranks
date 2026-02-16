# AI‑Stock‑Ranks – Agent Guide

This document serves as a concise table of contents and constraint list for agents operating in this repository.  Agents should read this file first before accessing deeper documentation in `docs/`.

## Purpose

The goal of this project is to produce a stable ranking of stocks by comparing them relative to each other in small batches.  A language model is used to perform listwise ranking within each batch.  The ratings from multiple independent batches are averaged to yield a final ranking that mitigates batch bias and context dilution.

## Directory overview

- `docs/PRD.md` – product requirements: describes inputs, outputs and non‑negotiable rules.
- `docs/DATA_CONTRACTS.md` – strict schemas for request and response payloads.
- `docs/PROMPTS.md` – templated prompts for the ranker and critic models.
- `docs/EVAL.md` – evaluation criteria and stability checks.
- `src/` – Python implementation of batching, LLM calls and aggregation.
- `prompts/` – concrete prompt files referenced by the no‑code orchestrator.
- `eval/fixtures/` – sample data for testing.
- `outputs/` – run artefacts (not tracked in git).

## Output contract

Ranker models must return JSON conforming to the `BatchResponse` schema (see `docs/DATA_CONTRACTS.md`).  Aggregated outputs must conform to the `AggregateOutput` schema.  No narrative prose is allowed in responses; only the specified fields.

## Evaluation rule

An execution is considered successful when all of the following hold:

1. Each batch response is valid JSON and matches the schema.
2. Every ticker appears in the final ranking exactly once.
3. Rank variance across trials remains below the threshold specified in `docs/EVAL.md`.
4. Stability tests on `eval/golden_sets/` succeed.

## Stop condition

Agents must continue refining prompts, data contracts and workflows until the evaluation checklist in `docs/EVAL.md` is fully satisfied.  Only then should a run be considered complete.