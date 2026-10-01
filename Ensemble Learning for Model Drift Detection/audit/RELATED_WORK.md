# Phase 3 — Related work and bibliography

Every entry in `paper/references.bib` was checked against Crossref by title
(`audit/verify_bib.py`) and, where the title match was ambiguous, by DOI lookup.
Result: all 27 entries resolve to a real record; **no fabricated references**.
Two gaps and two incomplete entries were found and fixed.

## Verified: existing entries resolve

`audit/verify_bib.py` output (title, DOI, year) confirms each entry. Three
ambiguous title matches were resolved by direct DOI lookup:

| key | correct DOI | verified |
|-----|-------------|----------|
| `gomes2017adaptive` | 10.1007/s10994-017-5642-8 | Adaptive random forests …, Machine Learning 2017 |
| `lu2018learning` | 10.1109/TKDE.2018.2876857 | Learning under Concept Drift: A Review, Lu et al. 2018 |
| `gama2004learning` | 10.1007/978-3-540-28645-5_29 | Learning with Drift Detection, SBIA 2004 |
| `pan2010transfer` | 10.1109/TKDE.2009.191 | A Survey on Transfer Learning, Pan & Yang 2010 |
| `ugr16` | 10.1016/j.cose.2017.11.004 | Maciá-Fernández et al., Computers & Security 73:411–424, 2018 |

## Gap 1 — recurring-concept literature is missing

The paper's whole thesis is that regimes recur, and it names "recurring concepts"
(line 95) and STAGGER/Widmer (lines 133–135), but it never cites the two works
that define the recurring-concepts problem directly. Both are standard and
verified:

- Widmer & Kubat, *Learning in the presence of concept drift and hidden
  contexts*, Machine Learning 23(1):69–101, 1996. DOI 10.1007/BF00116900.
- Katakis, Tsoumakas & Vlahavas, *Tracking recurring contexts using ensemble
  classifiers*, Knowledge and Information Systems 22(3):371–391, 2010.
  DOI 10.1007/s10115-009-0206-2.
- Gama & Kosina, *Recurrent concepts in data streams classification*,
  Knowledge and Information Systems 40(3):489–507, 2014.
  DOI 10.1007/s10115-013-0654-6.

Added as `widmer1996learning`, `katakis2010tracking`, `gama2014recurrent`.

## Gap 2 — dataset entries were incomplete

`nordicdat` and `nrlatency` were bare `@misc` records ("Zenodo record N,
Accessed 2026") with no authors, year, or DOI, so they render in the
bibliography as anonymous. Real metadata (Zenodo API):

| key | title | creators | date | DOI |
|-----|-------|----------|------|-----|
| `nordicdat` | NordicDat: A Cross-Border Predictive QoS Dataset | Miekkala, Pyykönen, Drainakis et al. | 2024-04-12 | 10.5281/zenodo.10964584 |
| `nrlatency` | 5G NR End-to-End Latency Simulation Dataset | Pakuła, Kryszkiewicz | 2026-05-05 | 10.5281/zenodo.20035549 |

The NordicDat dataset also has a peer-reviewed companion paper, which is the
citation a reviewer would expect:

- Miekkala et al., *NordicDat: A Cross-Border Predictive QoS Dataset*,
  IEEE GLOBECOM 2024, pp. 1281–1286. DOI 10.1109/GLOBECOM52923.2024.10901462.

`nordicdat` now points at the GLOBECOM paper (with the Zenodo DOI in a note);
`nrlatency` now carries its creators, year, and DOI.

## Changes applied to `paper/references.bib`

1. `nordicdat` → `@inproceedings` GLOBECOM 2024 with authors, pages, DOI.
2. `nrlatency` → `@misc` with creators, year, and Zenodo DOI.
3. Added `widmer1996learning`, `katakis2010tracking`, `gama2014recurrent`.
4. Added `doi` fields to `ugr16` and the three new recurring-concept entries.

No citation key already used in `main.tex` was renamed or removed, so the
existing text compiles unchanged. The three new keys are available for the
author to cite in Related Work §Transfer under Recurring Conditions (adding a
`\cite` changes prose, which is out of scope for this audit).

## Note on reference rendering

`paper/main.bbl` was generated with a style that omits DOIs, so the DOI fields
above will not appear until the bibliography is regenerated. The bib is the
source of record.
