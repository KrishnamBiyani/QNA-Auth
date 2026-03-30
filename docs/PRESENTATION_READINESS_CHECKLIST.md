# QNA-Auth: Exact Completion Playbook (Now -> Presentation)

This document is the literal execution plan you follow from this moment until presentation day.

It is written as operational instructions, not a high-level checklist.

Rules:
- Do steps in order.
- Do not move to the next phase unless the exit criteria are satisfied.
- If any step fails, log why, fix it, and re-run that step before continuing.
- Every claim you plan to present must have an artifact file proving it.

---

## Phase A - Session setup and control sheet (Start now, 30-45 min)

### A1) Create one tracking note before touching anything else
Create a single note document (local, anywhere) called `capstone_execution_log`.

Record these exact fields:
- Date/time started
- Current branch name
- Who is executing
- Dataset location used
- Intended presentation date
- Risk notes

### A2) Define your run ID convention
Decide one run ID format and stick to it (example: UTC timestamp style).
Every result folder, screenshot, and slide reference must include this run ID.

### A3) Establish your evidence folders
Confirm these folders exist (or will be generated):
- dataset manifest location
- split artifacts location
- capstone evaluation artifacts location
- reproducibility artifacts location
- longitudinal artifacts location

### A4) Define “done” now (non-negotiable)
Write this in your log:
- I will not claim anything that is not visible in artifact files.
- I will not call QRNG a device identity source.
- I will clearly separate proven claims vs future work.

Exit criteria for Phase A:
- tracking note exists
- run ID convention decided
- evidence folder map written
- done rules written

---

## Phase B - Data integrity lock (Day 1, 2-4 hours)

### B1) Inventory all incoming participant data
For each participant collection package:
- mark source (folder/zip name)
- mark expected device name/id
- mark expected session label
- mark whether camera/microphone are present
- mark whether file is valid or corrupted

Do not ingest blindly. First inventory, then ingest.

### B2) Ingest in controlled batches
Ingest collections in small batches and log each batch:
- batch number
- inputs included
- count ingested
- warnings/errors seen

If any package fails:
- move it to a quarantine list
- do not lose the file name
- document exact failure reason

### B3) Build canonical manifest and quality gates
Refresh the dataset manifest after ingestion.
Open the manifest and manually verify:
- total samples
- unique devices
- source distribution
- malformed count
- missing raw file count
- quality gate pass/fail

### B4) Resolve quality gate failures before model work
For each failure in manifest:
- classify as recoverable vs unrecoverable
- recoverable examples:
  - missing path fix
  - bad folder name mapping
  - missing session label you can infer
- unrecoverable examples:
  - truly missing raw data
  - corrupted arrays

For unrecoverable data:
- remove from active dataset
- keep audit note
- update manifest again

Exit criteria for Phase B:
- manifest is current
- failure reasons are documented
- you can speak exact data counts without guessing

---

## Phase C - Split policy and leakage assurance (Day 1-2, 1-2 hours)

### C1) Choose one primary split policy for headline results
Pick exactly one as primary:
- device-held-out split, or
- session-held-out split

Do not present mixed headline metrics from different split philosophies.

### C2) Generate split artifacts and verify leakage checks
Produce split artifacts and inspect them manually:
- confirm each split has expected devices/samples
- confirm no sample appears in multiple splits
- confirm policy intention matches split output

### C3) Write split rationale in plain English (for presentation)
Write one paragraph in your tracking note:
- why this split is chosen
- what leakage it prevents
- what it still does not test

Exit criteria for Phase C:
- split artifact exists
- leakage checks passed
- rationale paragraph is written

---

## Phase D - Model training run (Day 2, 2-6 hours depending on hardware)

### D1) Run one canonical training configuration first
Use a single baseline training config with fixed seed and record:
- seed
- epochs
- data path
- split artifact path
- training start/end time

### D2) Verify model output validity
After run:
- confirm checkpoint exists
- confirm checkpoint has expected metadata (feature version, dimensions)
- confirm no accidental random/uninitialized fallback was used

### D3) Decide whether to rerun
Only rerun if one of these is true:
- run failed technically
- split mapping was wrong
- manifest changed materially
- obvious metric anomaly from known error

Do not run endless retries chasing nicer numbers.

Exit criteria for Phase D:
- one traceable model checkpoint
- run metadata logged
- no unresolved technical errors

---

## Phase E - Capstone evaluation and baseline table (Day 2-3, 1-3 hours)

### E1) Execute capstone evaluation package
Run the capstone evaluation flow and capture run ID.
Confirm presence of:
- results JSON
- ROC image
- PR image
- split artifact pointer

### E2) Extract exact headline values
From results JSON, write these exact rows into your note:
- Siamese: EER, FAR, FRR, threshold
- Raw-feature cosine baseline: same metrics
- Small MLP baseline: same metrics

### E3) Build final presentation table now
Create one clean table with strict columns:
- method
- EER
- FAR
- FRR
- threshold
- interpretation (one short line)

Do this now, not on presentation day.

### E4) Lock the run
Choose one evaluation run as “presentation run”.
Mark in note:
- run ID
- artifact paths
- reason this run is chosen

Exit criteria for Phase E:
- final baseline table complete
- presentation run locked
- all table values trace to one artifact file

---

## Phase F - Attack evaluation narrative (Day 3, 1-2 hours)

### F1) Extract attack metrics exactly
Record these from artifacts:
- replay success rate
- impersonation success rate
- synthetic-statistical success rate

### F2) Write bounded interpretation text (mandatory)
Write exactly three parts:
1. what attacks are covered
2. what attacks are not covered
3. what claim this supports and what claim it does not support

### F3) Add one “if challenged” response
Prepare one response sentence:
- “This is preliminary attack coverage under our bounded threat model; stronger adaptive adversaries are future work.”

Exit criteria for Phase F:
- attack table exists
- bounded interpretation written
- no overclaim wording remains

---

## Phase G - Longitudinal section (Day 3-4)

### G1) Decide track based on available data
If multi-session data exists:
- run longitudinal analysis
- generate summary artifact
- create one drift plot and one paragraph interpretation

If multi-session data does not exist:
- explicitly label longitudinal as “pilot/in-progress”
- include data collection protocol for next weeks
- do not present fake stability claims

### G2) Failure-mode notes
Document at least one plausible failure mode from your setup:
- microphone environment shift
- lighting/camera noise change
- low sample count instability

Exit criteria for Phase G:
- either real drift evidence exists, or in-progress scope is clearly declared

---

## Phase H - Reproducibility proof (Day 4, 30-60 min)

### H1) Run one end-to-end reproducibility execution
Run the one-command reproducibility pipeline once from clean-ish state.

### H2) Verify metadata completeness
Confirm run metadata includes:
- run ID
- data path
- seed
- command chain
- timestamp

### H3) Prepare one reproducibility slide line
Write one sentence:
- “A single command regenerates manifest, split-aware train/eval, and result artifacts.”

Exit criteria for Phase H:
- reproducibility artifact exists and is readable
- reproducibility sentence is ready for slide

---

## Phase I - Documentation consistency lock (Day 4-5, 1 hour)

### I1) Read docs as a reviewer would
Read in this order:
1. README
2. DATASET guide
3. future reference

### I2) Remove contradictions
Fix any mismatch between docs and artifact reality:
- commands must match actual scripts
- paths must exist
- claims must match metrics

### I3) Red-flag language purge
Search for and remove wording like:
- production secure
- formally secure
- QRNG identifies the device

Exit criteria for Phase I:
- zero contradiction between docs and generated artifacts

---

## Phase J - Slide deck construction (Day 5, 3-5 hours)

Build the deck in this exact order and fill each slide with artifact-backed content:

1) Problem and bounded thesis  
2) System pipeline  
3) Dataset + split policy  
4) Baseline comparison table  
5) Threshold and EER explanation  
6) Attack results  
7) Longitudinal drift (or pilot status)  
8) Limitations and out-of-scope  
9) Reproducibility evidence  
10) Conclusion and next steps

For each slide, add a small “evidence source” note with artifact file path.

Exit criteria for Phase J:
- each slide has evidence source path
- no claim without corresponding artifact

---

## Phase K - Demo and fallback script (Day 6, 1-2 hours)

### K1) Prepare live demo path
Define a strict 3-minute live sequence:
1. show existing enrolled devices
2. run authentication
3. explain output with threshold context

### K2) Prepare offline fallback path
If live demo fails, immediately switch to:
- results table screenshot
- ROC/PR images
- attack table
- reproducibility metadata screenshot

### K3) Rehearse both paths once
Time both paths.
If either exceeds 3-4 minutes, simplify.

Exit criteria for Phase K:
- live and fallback both rehearsed once
- both can be delivered quickly

---

## Phase L - Defense prep and hard questions (Day 6-7, 1-2 hours)

Prepare concise answers (30-45 seconds each) to:
- Why QRNG is not device identity in your final claim
- How your split policy avoids leakage
- Why the baseline comparison is fair
- What attacks you tested and did not test
- Why your threshold is not arbitrary
- What is proven now vs future work

Then do one mock Q&A round with a friend/peer and note weak responses.

Exit criteria for Phase L:
- you can answer all six without improvising

---

## Final Go/No-Go Gate (Presentation morning)

You are “GO” only if all are true:

1. dataset manifest exists and is current  
2. split artifact exists for presentation run  
3. results JSON exists for presentation run  
4. baseline table in slides matches results JSON exactly  
5. attack table in slides matches results JSON exactly  
6. longitudinal section is honest (real evidence or explicitly in-progress)  
7. reproducibility metadata exists  
8. no slide contains unsupported claim language  

If any item is false, you are “NO-GO” until fixed.

---

## What “100% complete” means for this capstone

It does not mean perfect security.
It means:
- bounded thesis
- reproducible evidence
- fair baseline comparison
- leakage-safe evaluation
- transparent limitations
- presentation-ready narrative backed by files

If you meet all gates above, you are ready to present confidently.
