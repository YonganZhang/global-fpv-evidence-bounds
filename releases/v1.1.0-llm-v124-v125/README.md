# Actual FPV LLM experiment archive — v1.1.0-llm-v124-v125

This release supplements the earlier v118 repository. Its scientific reference is
the completed v124 primary/audit scoring experiment and v125 Jin-adapted water
analysis used by the September 2026 manuscript. It is not a rerun, a new set of
scores, or evidence of predictive accuracy. Older root-level v118 potential labels,
results and author metadata do not describe this release.

## What is included

- 4,701 accepted primary SDK records and 471 accepted comparison records, with
  **exact saved jobs, final-response strings and parsed outputs**. Eighteen rejected
  attempts are retained separately to document retries. Neither accepted nulls nor
  low-confidence judgments were resampled for this release.
- Four task protocols: ecology, maintenance (stored dimension `engineering`),
  national (`country`, with support and financing), and provincial (`province`).
  Each contains the original common system text, user template, context field order
  and output schema. Substituting the recorded context reconstructs every saved job.
- Original source snapshots for prompt generation, SDK adapters, retry/validation,
  score assembly, L2/L3 mappings and downstream calculations. Snapshot hashes match
  the historical records; the source has not been rewritten for retrospective claims.
- The frozen group catalogue and full 199,976-row internal-waterbody-ID mapping,
  derived score reference tables, comparison pairs and a real China waterbody example.
- Frozen **aggregate** present/future output tables and an offline entry rebuilding
  the headline ratios. No WDPA/LandScan site flags, source rasters or restricted
  per-waterbody potential outcomes are redistributed.

`response.raw_events` is omitted from each public record because it can contain
internal reasoning and duplicate event messages. Original record hashes and the
explicit omission paths are in `provenance/record_lineage.json`. Final answers,
requested reasons, jobs, scores, timestamps and recorded model metadata are unchanged.
These are redacted exports, not byte-identical copies of the original record files.
We do not request, publish or require internal chain of thought.

## Models and actual calls

Accepted answers were recorded from 8–9 September 2026 UTC. The primary requested
model was `gpt-6-astra`; its resolved identity was not independently observed.
The comparison SDK reported `claude-opus-5[1m]` in its initialization events.
Each job requested one turn and high effort. Temperature, top-p, seed and maximum
output tokens were **not specified in the actual SDK invocation**. The older API
constructor's temperature=0.2 and max_tokens=900 were not forwarded to the SDK.
See `protocol/runtime_parameters.json` for exact versions and transport settings.

The research policy was passed as Codex base instructions or Claude system_prompt.
Codex additionally received the developer instruction “Do not call tools. Return
the requested JSON only.” Accepted records contain no active tool items. This does
not prove that all tool capabilities were absent; `tools_exposed_empty_verified`
remains false. No external search or policy-text retrieval is added to this account.

## Offline reproduction

Install the packages in requirements.txt, then run from this directory:

```bash
python validate.py --report ../llm-release-validation.json
python reproduce.py --out ../llm-reconstruction
```

Neither command invokes a model or requires model credentials. The first checks
file hashes, membership, every request template, parsed final answer, retry history
and publication-field restrictions. The second uses the unchanged production
score-assembly function to reconstruct all waterbody scores exactly, checks the
comparison pairs, applies the original mapping kernel to the real example's scores,
and rebuilds the paper's percentages from frozen aggregate tables.

The example uses actual input/response records but demonstrates **score-to-weight
mapping only**. It does not invent L1 generation, protected status, costs or project
viability for the waterbody. Country/province names and ecoregion tags are the actual
coarse prompt context, not validated site surveys.

## Three distinct levels of reproducibility

1. Final answers → scores → waterbody mapping and weighting: fully reproducible here.
2. Frozen aggregate outputs → paper headline ratios and climate ranges: reproducible
   here, including L3/L1 water retention of 6.41–15.14% versus L3/L2 of 17.83–37.19%.
3. Provider data → physical/geospatial potential → global outputs: requires separately
   authorized source data. This archive does **not** claim a self-contained end-to-end
   rebuild of these restricted-data steps. `provenance/numerical_contract_original.json`
   records the exact input paths/hashes; the unchanged implementation is under
   historical_source/src/analysis/lean_global_consumers_v124.py and its dependencies.
   The v125 water entry is historical_source/src/analysis/run_jin_water_v125.py.

To reconstruct the complete original run, restore the authorized inventory, enriched
tags, all physical partitions and original run-history artifacts at the paths in the
contracts, then use the historical consumers' `inputs`, `levels`, `future` and `benefits`
entries followed by the v125 water runner. Use new output run IDs. Do not launch the
SDK schedulers by default and do not label new model answers as the original experiment.
Historical integrity validators can require omitted intermediate receipts and raw
events; original-record hash verification needs the private originals, whereas the
portable public validator checks the explicitly exported subset.

## Interpretation

LLM scores are pretrained-knowledge judgments, not missing observations, verified
policy measurements or calibrated feasibility probabilities. Researcher-defined
mappings affect ecological area, O&M cost only, and development-area weights. The
comparison model is diagnostic and never fills or averages primary scores. Contemporary
judgments and other non-climate development conditions remain fixed in future scenarios.
Future candidate changes include changes in ice-information support, not only physical
thaw. Avoided evaporation is not delivered water; annual generation ratios are not
hourly system adequacy. See DATA_LICENSES.md before reusing third-party inputs.

No DOI has been minted for this version. Cite the versioned GitHub release and its
checksums until a genuine archive DOI is available; never substitute a reserved or
invented DOI. This release is not all-author approval of a journal submission.
