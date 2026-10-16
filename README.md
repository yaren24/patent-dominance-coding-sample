# Dominant inventors and their fields: a coding sample

Does the death of a technology field's leading inventor change how much is invented in that
field? My MA thesis studies this question with US patent data, in the spirit of Azoulay,
Fons-Rosen and Graff Zivin (2019) on the deaths of star scientists.

This code is taken from my MA thesis, which is in progress. It is a snapshot of two steps of
the full pipeline:

| File | What it does |
| --- | --- |
| `build_dominance.py` | **Part A, data construction.** From the raw PatentsView tables to each inventor's share, rank and dominance margin in every technology field and year. |
| `estimate_event_study.py` | **Part B, estimation.** A stacked Poisson event study of the field's patenting before and after its leader's death, with two-way clustered standard errors and a pre-death test. |
| `tests/` | Tests on small made-up data with known answers. |

The steps between the two parts (death records, matching each deceased leader's field to
control fields, building the outcome panel) are not included. No results are reported here.

I designed the research and decided every rule in this code. I wrote the thesis pipeline with Claude Code, an AI coding assistant, and reviewed and tested each step. This sample was cut from that pipeline.

## Data

Part A uses four public tables from the **PatentsView** granted-patent bulk data (USPTO,
public domain), downloaded from <https://data.uspto.gov/> (PatentsView download tables):

- `g_patent.tsv`: patent type, withdrawn flag
- `g_application.tsv`: application (filing) date
- `g_inventor_disambiguated.tsv`: the inventors listed on each patent
- `g_cpc_current.tsv`: CPC technology codes

The death dates behind Part B's input come from Kaltenberg, Jaffe and Lachman, "The Age of
Invention" (NBER WP 28768; data on Harvard Dataverse, <https://doi.org/10.7910/DVN/YRLSKU>).
Part B's input panel is described at the top of `estimate_event_study.py`.

## How to run

Python 3.11 or newer.

```bash
pip install -r requirements.txt

# Part A: unzip the four tables into data/patentsview/, then
python build_dominance.py --data-dir data/patentsview --out-dir output

# Part B, given an outcome panel built by the thesis's earlier steps
PYTHONHASHSEED=0 python estimate_event_study.py --panel outcome_panel.parquet

# Tests
pytest
```

The raw tables are large. DuckDB keeps memory under `--memory-limit` (2 GB by default) and
spills to disk above it. On the full data, Part A took about four minutes on a laptop and, at
the default limit, wrote about 27 GB of temporary files, so leave about 30 GB of disk free
(a higher limit spills less). The temporary files are deleted when the run ends.

## Part A: measuring dominance

- **Dates.** A patent is dated by its application year, when the invention was made, not its
  grant year, which adds an examination delay that varies by technology and decade.
- **Universe.** Utility patents, not withdrawn, filed 1976 to 2022.
- **Fields.** A field is the CPC main group of the patent's primary code (for example
  A61K31/00).
- **Fractional counting.** A patent with n inventors gives 1/n to each. Shares within a field
  then sum to one, and nobody becomes dominant only by working in large teams.
- **The window.** An inventor's share in year t uses the patents filed from t-4 to t. The window
  is built forwards from each patent's filing year, so a patent can never enter a window that
  closed before it was filed.
- **Rank and margin.** Inventors with equal weight share a rank. The margin is the leader's
  weight divided by the runner-up's: a tied top has a margin of exactly 1, and a field with one
  inventor has no margin rather than an invented one. Weights are rounded before ranking, so
  that sums like 1/2 + 1/3 + 1/6 and 1/2 + 1/2 tie as they should.
- **Checks.** The run stops before writing anything unless shares sum to one in every
  field-year, every share lies in [0, 1], every field-year has a leader, every margin is at
  least one, keys are unique, and the total credit is unchanged by the windows. Every dropped
  row is counted and its reason printed.

## Part B: the event study

- **Stacks.** Each deceased leader's field and its matched control fields form a stack,
  observed from five years before her death (D-5) to five years after (D+5).
- **Model.** Weighted Poisson regression of the field's yearly patent count on treated x year
  terms, with fixed effects for each field within its stack and for each stack and year. The
  event study measures each year against D-1. The pooled version compares D+1 to D+5 with
  D-5 to D-1. Effects are read as exp(b) - 1.
- **Weights.** Each death counts once on each side in every year, however many fields she led
  and however many controls each field has.
- **Inference.** Standard errors are clustered two ways, by the deceased and by the field,
  because one person can lead several fields and one field can be a control in several stacks.
  The pre-death test is a joint Wald test that D-5 to D-2 are all zero.
- **Guards.** Rows that carry no information (fields or stack-years with no patents at all, or
  left with one row) are removed before the fit. A year in which no treated field has a patent
  stops the run, because its coefficient would otherwise be reported as a plausible number. The
  fit also stops if it does not converge, drops a term or a row, or reports a variance that is
  not positive.

## Tests

- **Part A, on a hand-built extract of nine patents:**
  - shares sum to one;
  - the window looks only backwards;
  - a tie at the top has a margin of one;
  - patents outside the universe give no one credit;
  - a person listed twice on a patent is one inventor;
  - putting the wrong denominator into the share stops the build.
- **Part B, on simulated panels with a known 20% fall after the death:**
  - the fit recovers the fall and finds no pre-trend;
  - the coefficients equal statsmodels' with every fixed effect written out as a dummy;
  - a year with no treated patents stops the fit;
  - the Wald test equals a hand computation.
