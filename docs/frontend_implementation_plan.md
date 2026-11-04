# UpliftIQ frontend implementation plan

Status: approved-for-build blueprint (planning only; no code yet). Backend baseline: commit `c701f4a`.
Companion to `docs/frontend_data_contract.md`. Where this plan and the contract differ on route
names, the **implemented backend** (`src/upliftiq/backend/`) wins; paths below are the real ones.

Ground rules for every screen:
- Every number comes from an API response. No literal results in TSX, no fixtures in production code.
- Hillstrom and Criteo never share an axis, a ranking, a KPI row or a color meaning.
- Criteo is never shown in currency and never described as ROI or population impact.
- Each chart and KPI carries its evidence type (held-out / validation / cross-validated / live model
  output / stored offline measurement) and a source footnote (`envelope.source`).
- Where no data exists (e.g. gain-curve points, budgets other than 10/20/30%), say so; never interpolate.

---

## A. Design system

### A1. Color tokens (CSS variables in `app/globals.css`, mapped in Tailwind v4 `@theme`)

| Token | Value | Use |
|:--|:--|:--|
| `--bg` | `#F5F7FA` | app background |
| `--surface` | `#FFFFFF` | panels, cards |
| `--surface-subtle` | `#EEF1F5` | table headers, inset areas, segmented-control track |
| `--border` | `#E2E7EE` | panel borders, dividers |
| `--border-strong` | `#CBD3DD` | inputs, focusable outlines at rest |
| `--text` | `#0F1726` | primary text |
| `--text-secondary` | `#4A5568` | descriptions, axis titles |
| `--text-muted` | `#6B7686` | captions, footnotes |
| `--sidebar` | `#121B2C` | sidebar background |
| `--sidebar-elevated` | `#1A2538` | hover/active row |
| `--sidebar-border` | `#26324A` | sidebar dividers |
| `--sidebar-text` | `#C5CEDB` / active `#FFFFFF` | nav text |
| `--teal` | `#117E74` | primary accent, primary buttons, **held-out / selected** evidence |
| `--teal-soft` | `#E3F2F0` | teal fills, selected rows |
| `--violet` | `#7467C9` (text: `#5E51B5`) | secondary accent, **validation / model-selection** evidence |
| `--violet-soft` | `#ECEAF8` | |
| `--amber` | `#D99A43` (text: `#8F5D17`) | **caveats, exploratory, warnings** (fills/icons only; text uses the dark variant) |
| `--amber-soft` | `#FBF1E2` | caveat callouts |
| `--danger` | `#C2413B` / soft `#FBEAE9` | errors, failed checks |
| `--ref` | `#64748B` | baselines/references (random, treat-all, no-email) |
| `--ref-soft` | `#94A3B8` | dashed reference lines |

Semantic rule: **color encodes evidence type, not dataset.** Datasets are identified by a labelled
chip with an icon (Hillstrom: `Mail`, Criteo: `FlaskConical`), never by hue.

Model series palette (fixed mapping, `lib/colors.ts`, identical in every chart):
`s_learner #117E74`, `tarnet_cpu #7467C9`, `tarnet_mps #A99FE0`, `logistic #3E7CB1`,
`t_learner #B8607A`, `dr_learner #7F8F4E`. Hillstrom policies: `mens_for_all #117E74`,
`womens_for_all #7467C9`, `no_email #64748B`, `random #94A3B8`, CV models use the model palette.

Light theme only in v1; tokens are defined so a dark theme can be added later without refactoring.

### A2. Typography
- Fonts: **Geist Sans** and **Geist Mono** via `next/font` (no runtime font requests).
- `font-variant-numeric: tabular-nums` on `body`; mono only for hashes, ids, file paths, config keys.

| Role | Size/line | Weight | Notes |
|:--|:--|:--|:--|
| Page title | 24/32 | 600 | tracking -0.01em |
| Section title | 16/24 | 600 | |
| Panel title | 14/20 | 600 | |
| Body | 14/20 | 400 | |
| Small / table | 13/18 | 400 | |
| Caption / footnote | 12/16 | 400 | `--text-muted` |
| Overline | 11/16 | 500 | uppercase, tracking 0.06em, `--text-muted` |
| KPI value | 28/32 | 600 | tabular; CI underneath at 12/16 |

### A3. Spacing, surfaces, radii, elevation
- 4 px base. Page padding: 32 (≥1280), 24 (768–1279), 16 (<768). Panel padding 20 (16 on mobile).
  Grid gap 16; section gap 32. Max content width 1440, centered.
- 12-column grid (`grid-cols-12 gap-4`); spans used: 12, 8/4, 7/5, 6/6, 4/4/4, 3×4.
- Surfaces (3 levels only): page `--bg` → panel `--surface` + 1 px `--border` + `shadow-xs`
  (`0 1px 2px rgb(16 24 40 / .04)`) → inset `--surface-subtle`. Popovers/sheets: `0 8px 24px rgb(16 24 40 / .12)`.
- Radii: panels 10, controls 8, chips/badges 6. No gradients, no glassmorphism.

### A4. Navigation shell
- **Sidebar** (dark, shadcn `Sidebar`): 248 px expanded / 64 px collapsed (icon rail with tooltips).
  - Top: UpliftIQ wordmark + "Causal decision intelligence" subline.
  - Group "Workspaces" with numbered items: `01 Overview` (`LayoutDashboard`), `02 Experiment Lab`
    (`FlaskConical`), `03 Uplift Explorer` (`ChartSpline`), `04 Decision Studio` (`Target`),
    `05 Model Operations` (`ServerCog`). Active: `--sidebar-elevated` row + 2 px teal left bar + white text.
  - Footer: live API status (dot + "API online · model 1.0.0" / "Degraded" / "Offline") from
    `/health`; link "API reference" → backend `/docs`; collapse toggle (`⌘/Ctrl+B`).
  - Collapse state persisted in a cookie (shadcn default) so SSR renders the right width.
  - ≥1280 expanded by default; 768–1279 collapsed by default; <768 hidden, opened as a `Sheet` from a
    hamburger in the top bar.
- **Top bar** (56 px, `--surface`, bottom border, sticky): breadcrumbs
  (`UpliftIQ / Decision Studio / Criteo`), **dataset switch** (segmented control `Hillstrom | Criteo`)
  when the workspace supports both, an evidence chip describing what the page shows, and page actions
  (right-aligned, e.g. "Export CSV", "Refresh status").
- **Page header** (inside content): overline (workspace number + name), title, one-sentence
  description, dataset chip, optional actions. Then content sections.

### A5. Metric formatting (`lib/format.ts`; the only place numbers are formatted)

| Quantity | Format | Example |
|:--|:--|:--|
| Hillstrom rate (proportion) | percent, 1 dp | `18.2%` |
| Hillstrom rate difference | pp, 2 dp, signed | `+2.94 pp` |
| Hillstrom spend | USD, 2 dp | `$1.42` per customer |
| Criteo policy value / gain (stored as fraction) | ×100, pp, 3 dp | `0.674 pp` |
| Criteo predicted probability | percent, 2 dp | `5.28%` |
| Criteo predicted uplift | pp, 3 dp, signed | `+1.291 pp` |
| Interval | value then `[low, high]`, same unit/dp; label "95% CI" in tooltip | `0.674 pp [0.616, 0.715]` |
| p-value | `p < 0.001` below 0.001, else 3 dp; Holm-adjusted labelled "Holm" | `p (Holm) < 0.001` |
| Counts | full with separators in tables; compact in KPIs with full value in tooltip | `2,795,918` / `2.80M` |
| Duration | `< 1 ms` → 2 dp ms; ms 1 dp; seconds 1 dp | `0.33 ms`, `48.3 s` |
| Memory | GB, 2 dp | `2.15 GB` |
| Hash | mono, `first8…last6`, copy button, full on hover | `1e5059cb…65dcf980` |
| Budget | percent, integer | `20%` |

Separate typed formatters (`hillstromRate`, `criteoPp`, …) so Criteo values cannot be passed to a
currency formatter. Signs: always show `+`/`−` on differences. Unicode minus `−`.

### A6. Chart rules (Recharts; shared in `components/charts/`)
- Every chart sits in `ChartFrame`: title, subtitle with **units and evidence type**, legend,
  "View as table" toggle (accessible data table of exactly the plotted values), footnote with source
  report paths and provenance badge.
- Horizontal gridlines only (`#E7EBF0`), no vertical gridlines; axis text 12 px `--text-muted`;
  axis titles 12 px `--text-secondary`; zero line 1 px `--border-strong`.
- Uncertainty is always drawn when it exists: whiskers (`ErrorBar`) for points, light bands for
  series. Point estimates without their CI are not allowed in charts.
- **No interpolation between evaluated budgets.** Budget charts show markers + whiskers at 10/20/30%
  only; references (treat-nobody 0, treat-all) are horizontal labelled lines/bands, not curve points.
- Reference/baseline series: `--ref`, dashed (`4 3`), labelled inline at the line end.
- Selected/primary series: 2 px stroke or filled marker; others 1.5 px / hollow markers; non-selected
  models when comparing fade to 30% opacity on hover-focus of another.
- Custom tooltip (`ChartTooltip`): series name, value + CI, n, evidence type. White panel, 1 px border,
  popover shadow, tabular numbers.
- Stored PNG figures (only where no numeric data exists, e.g. held-out gain-curve shape) render in
  `ReportFigure`: framed image, caption "Stored report figure (static)", source path, click to enlarge.
- Animation: 300 ms ease-out on mount only; `isAnimationActive={!prefersReducedMotion}`.
- Heights: 320 primary, 240 secondary, 200 on mobile. `ResponsiveContainer` everywhere.
- Never place Hillstrom and Criteo values on one chart.

### A7. Interaction patterns
- Selectors: segmented control (≤4 options, e.g. outcome, evidence scope), `Select` (more options),
  multi-select chips for model comparison (max 3), `Slider` + numeric input for budget.
- Selections that define a view (dataset, outcome, scope, compared models) live in the URL query
  (`?dataset=criteo&outcome=visit&models=s_learner,tarnet_cpu`) so views are shareable and survive
  reload. Ephemeral UI state (open accordion, hover) is local.
- Explanations: `Tooltip` on `Info` icons for definitions (AIPW, Qini, Holm, one-SE rule);
  `Accordion` "Methodology" sections for longer text; `CaveatCallout` (amber) for limitations, never
  collapsed by default where the contract says "show, do not hide" (IPW/AIPW).
- Badges: `Supported` (teal, Holm p < 0.05 on the primary outcome), `Exploratory` (amber),
  `Held-out` (teal outline), `Validation` (violet outline), `Cross-validated` (violet outline),
  `Live` (teal dot), `Stored measurement` (slate), `Synthetic` (amber, solid).
- Motion: hover/press 150 ms, panel/drawer 200 ms, skeleton→content 150 ms fade. Respect
  `prefers-reduced-motion` (Tailwind `motion-reduce:`).
- Focus: 2 px teal ring, 2 px offset, visible on every interactive element; full keyboard path through
  sidebar, top bar, selectors, slider, tables and drawers. Minimum text contrast 4.5:1 (use the dark
  text variants of violet/amber).

### A8. Responsive behavior
- ≥1280: full 12-column layouts as specified per page.
- 1024–1279: sidebar collapsed; 8/4 splits become 7/5 or stack where noted.
- 768–1023: two-column grids become single column except KPI rows (2-up); Decision Studio stacks
  configuration above results.
- <768: single column; KPI 2-up; wide tables scroll horizontally **inside** their panel (no page
  scroll); leaderboard becomes a card list; charts 200 px high; top-bar dataset switch moves into the
  page header.

### A9. Standard states (`components/states/`)
- `PanelSkeleton` (matches the final panel's height to avoid layout shift), `KpiSkeleton`, `ChartSkeleton`.
- `ReportUnavailable` (404 `report_unavailable`): "This report isn't available in this deployment",
  shows the missing path from `detail`; rest of the page keeps working.
- `ModelUnavailable` (503 `model_unavailable`): disables run controls, explains the model package is
  not loaded, links to Model Operations.
- `ApiOffline` (network error): retry button; top-bar status turns "Offline".
- `ValidationErrors` (422 `invalid_request` / `invalid_input`): field-level messages from `detail[].loc/msg`.
- `EmptyState`: icon, one sentence, primary action.

---

## B. Page layouts

Routes: `/` redirects to `/overview`; `/overview`, `/experiments`, `/uplift`, `/decisions`, `/operations`.

### 01 Overview — Executive intelligence (`/overview`)
Data: `/overview`, `/datasets`, `/health`, `/hillstrom/decision` (policy values), `/criteo/heldout`.

Header: overline "01 Overview", title "Decision intelligence overview", description "Two randomized
experiments, two frozen decisions, each evaluated once on held-out customers." Dataset switch selects
the **briefing** shown (default `hillstrom`); the other study stays visible as a compact card.

1. **Briefing KPI row** (4 × span-3):
   - Hillstrom: Frozen decision ("Men's E-Mail for all"); Held-out visit rate `18.2% [17.1, 19.3]`;
     vs strongest alternative (Women's for all) `+2.94 pp [ci]`, `p (Holm)`; Held-out customers `12,800`.
   - Criteo: AIPW value @20% `0.674 pp [0.616, 0.715]`; vs random @20% `+0.525 pp [ci]`;
     Treat-all reference `0.744 pp [ci]` (labelled "unconstrained, 100% budget"); Test customers `2.80M`.
2. **Primary visualization** (span-8):
   - Hillstrom: horizontal dot-and-whisker of held-out AIPW visit rate for `no_email`, `random`,
     `womens_for_all`, `mens_for_all` (frozen highlighted teal).
   - Criteo: `BudgetValueChart` — markers + CI at 10/20/30% for S-learner, random markers at the same
     budgets, treat-all as a dashed band with CI, zero baseline. Title "Held-out policy value by
     contact budget"; footnote "Only these budgets were evaluated."
3. **Decision insight panel** (span-4): "The decision" — 3–4 sentences built from data (frozen policy
   or model, primary metric, comparison, why it was chosen) + caveat list from `envelope.caveats`
   (amber callout for Criteo with "Benchmark, not business ROI" label, required by the contract).
4. **Other study card** (span-4) with its single headline KPI and "Open briefing" link; plus
   **Platform status** (span-4): API status, model version/loaded, reports available
   (`/health.reports_available`); and **Evidence trail** (span-4): evaluation record status, one-shot,
   code commit, `code_dirty: true` shown as recorded.
5. **Continue the analysis** row (4 cards linking to workspaces 02–05 with one-line purpose each,
   query string preserving the dataset).

States: KPI/Chart skeletons per section; a failing endpoint degrades only its section.

### 02 Experiment Lab — Evidence & statistics (`/experiments`)
Data: Hillstrom `/hillstrom/groups`, `/hillstrom/effects`, `/hillstrom/balance`, figures;
Criteo `/criteo/diagnostics`, `/criteo/ipw-aipw`, figure `treatment_balance`.

**Hillstrom variant**
- Controls (sticky sub-bar): Outcome `Visit (primary) | Conversion | Spend`, Sample
  `Full sample (64,000) | Held-out (12,800)` (maps to `effects.full_sample` / `effects.heldout`).
- Row 1: `ArmOutcomeBars` (span-7): outcome by arm from `groups` (rate or mean spend) with n in the
  tooltip; `ArmSummaryTable` (span-5): n, share, visit/conversion rate, mean spend, share zero spend.
- Row 2: `ForestPlot` (span-8): absolute effect with CI for each comparison on the selected outcome;
  toggle Absolute/Relative; each row badged `Supported` (Holm-significant, primary outcome) /
  `Significant (exploratory)` / `Not significant`; p and Holm p in the table view.
  `EffectReadout` (span-4): plain-language reading of the selected comparison and method used
  (`method` column: z-test/Newcombe, bootstrap/permutation).
- Row 3: `BalanceDotPlot` (span-8): `max_abs_smd` per covariate for Men's and Women's vs control
  with a 0.1 threshold line; `BalanceTable` (span-4): test, p, balanced flag.
- Row 4: stored figures (spend distribution, covariate distributions) in `ReportFigure` (span-6 each),
  since those tables aren't served.
- `Methodology` accordion: randomization, primary vs exploratory outcomes, Holm correction, why spend
  uses bootstrap/permutation, held-out vs full-sample interpretation.
- Note: significance differs by sample (e.g. on the held-out sample Men's-vs-none conversion and spend
  are not Holm-significant, while Women's-vs-none spend is); badges come from
  `significant_holm_0.05` and the outcome role, never hardcoded. Exploratory outcomes stay labelled
  `Exploratory` even when significant.

**Criteo variant**
- Row 1 KPIs (4 × span-3): rows `13.98M`, treated share `85.0%`, split sizes (train/val/test),
  verdict `benchmark_modeling_allowed`.
- Row 2: `RandomizationChecks` (span-5): list of checks A1a–A2 with value, rule, grade chip
  (`pass` teal, `warn` amber, `ordering_warning` amber outline) — the warnings stay visible;
  `SmdDotPlot` (span-7): SMD for f0–f11 with ±0.05/0.10 thresholds; tooltip with KS.
- Row 3: `PropensityPanel` (span-6): classifier AUC (LightGBM, logistic) vs permuted-label AUC with SE;
  predicted propensity quantiles. `SplitPanel` (span-6): sizes, treated share per split, fingerprint (Hash).
- Row 4 **IPW vs AIPW (shown, not hidden)** (span-12, amber-bordered): `EstimatorComparison`
  dot-and-whisker of IPW, AIPW, g-computation, post-stratified (from `paired_intervals` and
  `poststratified_on_predicted_baseline`), in pp; `BaselineGap` readout (predicted baseline treated vs
  control, permutation z and p); `StrataChart` (10 strata of predicted baseline: treated share and
  effect). Text: AIPW primary, IPW cross-check, why they disagree, what it means.
- Observed arm visit rates are shown in a table labelled "Observed, unadjusted; released sample";
  no ROI or population framing.

### 03 Uplift Explorer — ML research workspace (`/uplift`) — flagship
Data: Criteo `/criteo/benchmark`, `/criteo/heldout`, `/criteo/curves`, figures; Hillstrom `/hillstrom/models`.
Default dataset `criteo`.

**Criteo variant**
- Sub-bar: **Evidence scope** `Validation (model selection, 500K) | Held-out (one-shot, S-learner only)`
  — violet vs teal accent on the whole content area's evidence chip; **Compare** model chips
  (max 3; default `s_learner, tarnet_cpu, logistic`); **Metric** `AIPW @10% | @20% (primary) | @30% | Qini`.
- Validation scope:
  1. `Leaderboard` (span-12): rows = full-scale finalists (logistic, s_learner, tarnet_cpu);
     columns: AIPW @10/20/30 with CI, vs random @20%, Qini AIPW with CI, IPW (muted, "cross-check"),
     train time, p50 latency @1, rows/s @100k, peak RSS; status badges `Selected`, `Best by value`,
     `Best by Qini`. Sort by any column; row click opens `ModelDetailSheet`.
  2. `BudgetValueChart` (span-7) for compared models; `PairwiseForest` (span-5): paired
     differences vs a chosen reference model (`pairwise`), whisker crossing zero greyed.
  3. `LearningCurve` (span-8): x = training rows on a log scale (100K, 500K, 2M, 8.9M), y = selected
     metric with CI band; 6 models through 2M, 3 finalists at full scale (full-scale point drawn as a
     filled diamond; models without a full-scale run simply end at 2M). `ScalingReadout` (span-4):
     "2M → full" change with CI per finalist (`change_from_2m`) and diminishing-returns statement.
  4. Engineering row: `TrainingTimeChart` (span-4, train seconds vs size, log-log),
     `CostQualityScatter` (span-4, train seconds vs Qini with CI), `LatencyBars` (span-4, p50/p95 at
     batch 1 / 1k / 100k for finalists). All labelled "Stored offline measurement (single machine,
     OMP_NUM_THREADS=1)".
  5. `SelectionRationale` (span-12): stepper rendered from `selection.order` and `selection.steps`:
     simplicity order, each step's paired difference vs best with CI and "within one SE?" outcome,
     chosen model, and "Qini supports choice" check. Trade-off text generated from the steps
     (logistic is more than one SE below TARNet; the S-learner is within one SE of TARNet at 20%, a
     difference of essentially zero, and is the simpler model, so it is chosen).
- Held-out scope:
  1. KPI row: value @20% (primary), vs random @20%, Qini AIPW, treat-all reference.
  2. `BudgetValueChart` (span-7) S-learner vs random with treat-all band; `HeldoutTable` (span-5)
     with IPW cross-check column muted and an IPW>AIPW note.
  3. `ReportFigure` (span-6) stored held-out gain curve (the only source of curve shape) and
     `ReportFigure` value-vs-random (span-6).
  4. `ExploratoryConversion` (span-12, collapsed accordion, amber "Exploratory, never used for any
     decision" badge).
  5. Validation-vs-held-out comparison table for the S-learner (same three budgets), labelled
     "different samples; not a paired comparison".
- `ModelDetailSheet` (right sheet): family, config key, hyperparameters, TARNet training (epochs run,
  best epoch, device), libraries, code commit + dirty flag, uplift mean/std.

**Hillstrom variant** (no curves exist; no served model)
- `CandidateForest` (span-8): cross-validated AIPW value with CI for 4 models + 4 fixed baselines
  (`selection_summary`), Men's-for-all highlighted; reference line at Men's-for-all value.
- `SelectionPanel` (span-4): rule ("one paired SE; simplest eligible; baselines may win"), best by
  value, eligible/ineligible models, chosen policy; difference vs Men's-for-all with CI per model.
- Note: "Cross-validated on the development set; the held-out result is in Overview / Decision Studio."

### 04 Decision Studio — Interactive targeting (`/decisions`) — signature screen
Default dataset `criteo`. Layout ≥1024: left config panel span-4 (sticky, scrolls independently),
results span-8. <1024: stacked.

**Criteo variant** (live model; `/criteo/target`, `/criteo/predict`, plus `/criteo/heldout`,
`/criteo/diagnostics` for evidence and the cohort generator, `/health` for availability)

Left panel — Decision configuration:
1. **Cohort source** (tabs):
   - `Synthetic demo cohort` (default): size (100–10,000, default 1,000) and seed (default 42).
     Generated client-side from the per-feature quantiles in `/criteo/diagnostics.data.dataset.features`
     (see C7). Amber `Synthetic` badge + one-line note: "Independently sampled from released feature
     marginals; not real customers; joint structure not preserved."
   - `Upload / paste`: CSV or JSON with columns `f0`–`f11` (+ optional integer `id`); client-side
     parse and validation (all 12 finite numbers, no extra columns, ids all-or-none and unique,
     ≤ 10,000 rows) with row/column error list before sending.
   - `Single customer`: 12 numeric inputs prefilled with feature medians, "Reset to medians"; calls
     `/criteo/predict` and shows the single-record card (no budget).
2. **Contact budget**: slider 1–100% (step 1) + numeric input; tick marks at 10/20/30% labelled
   "validated"; live readout "Contact k = ⌊budget × n⌋ = 200 of 1,000".
3. **Run targeting** primary button (disabled when model unavailable or input invalid; `⌘/Ctrl+Enter`).
   Shows request state; cancel on rerun (AbortController).
4. Footer: model version, ranking rule, link to Model Operations.

Right panel — Decision results (empty state before first run: illustration-free panel explaining
what will appear):
1. **Allocation summary** (4 KPIs): customers `n`, selected `k` (share), mean predicted uplift
   selected vs not selected, share of cohort with positive predicted uplift. Label: "Model predictions
   on your cohort; calibration not validated."
2. `UpliftRankChart` (span-12): predicted uplift sorted descending (x = rank percentile, y = uplift pp),
   selected region filled teal, cut-off line at k with label, zero line. This is a ranking of model
   outputs, not a gain curve; subtitle says so.
3. `UpliftDistribution` (span-6): histogram of predicted uplift, selected vs not selected overlay.
   `ProbabilityScatter` (span-6): p_visit_control vs p_visit_treatment (sampled to ≤ 2,000 points),
   selected colored.
4. `RankedTable` (span-12): rank, id, selected chip, p_control, p_treatment, uplift; filter
   All/Selected/Not selected; client pagination (50/page); "Export CSV" (client-side file from the
   response; synthetic exports carry a `synthetic=true` column).
5. `ValidatedEvidence` (span-6):
   - budget ∈ {10, 20, 30}%: held-out AIPW value at that budget with CI, vs random, from
     `/criteo/heldout`, captioned "Held-out estimate for this policy on the Criteo test set; it is not
     an estimate for your cohort."
   - other budgets: "No validated estimate at 37%. Held-out estimates exist only for 10%, 20% and 30%."
     with those three values shown for context. Never interpolated.
6. `HowThisWasDecided` (span-6): ranking string from the response, tie-break by seeded id hash
   (seed from manifest via `/models/criteo`), k rule, model version, response `caveats`.

States: `ModelUnavailable` replaces the run button area and results; 422 → field/row errors in the left
panel; network error → retry; > 10,000 rows rejected client-side with the API limit message.

**Hillstrom variant** (no served model; the decision is a fixed policy)
- Left panel: Outcome `Visit (primary) | Conversion | Spend`, Estimator `AIPW (primary) | IPW
  (cross-check)`, display unit `Rate | Per 1,000 customers` (a rescaling of the held-out rate,
  labelled "held-out sample estimate"). No run button; read-only callout: "Same email for every
  customer: no personalised model beat this fixed policy."
- Right panel: `PolicyBoard` — dot-and-whisker of held-out policy values (frozen = Men's for all,
  Women's for all, no email, random) for the selected outcome/estimator; `ComparisonTable` of paired
  differences (`comparisons`), primary row flagged, Holm p, `Supported`/`Exploratory` badges;
  `DecisionRecord` card (chosen policy, frozen config hash, held-out n, evaluation record); link to
  Uplift Explorer for the model-selection evidence.

### 05 Model Operations — ML engineering console (`/operations`)
Data: `/health` (no-store), `/models`, `/models/criteo`, `/models/criteo/runs` (may 404), `/criteo/benchmark`
(stored engineering), live probes via `/criteo/predict`.

1. **Status strip** (span-12): API status, model loaded + version, model error (if any), reports
   availability checklist (from `/health.reports_available`), "Refresh" action, last checked time.
   Labelled `Live`.
2. **Model card** (span-7): name, version, status, algorithm, created_at, artifact file/format,
   sha256 (Hash with copy), input schema table (f0–f11, dtype, constraints, preprocessing, treatment
   indicator note, max batch), outputs table. **Reproducibility** (span-5): frozen vs packaged
   validation uplift sha (match ✓), held-out test set read during packaging: false, dependencies.
3. **Lineage** (span-12): horizontal stepper (vertical on mobile): Dataset (version, sha, rows) →
   Split (fingerprint, seed, sizes) → Training (rows, seed, hyperparameters) → Validation selection
   (frozen config sha) → Held-out evaluation (one-shot, status, code commit, `code_dirty: true`
   as recorded, test ids/uplift sha) → Package (artifact sha) → Serving (this API). Each node opens
   its details in a popover.
4. **Training configuration** (span-6): hyperparameter key/value grid (mono keys).
   **Historical results** (span-6): validation vs held-out headline values, both labelled.
5. **Experiment tracking** (span-12): MLflow runs table (name, provenance badge `historical` /
   `new_execution`, executed_by_this_run, model version, start time, key metrics). When
   `/models/criteo/runs` returns 404: fallback panel "Local MLflow store not included in this
   deployment" + the same lineage facts from the committed manifest and evaluation record, and the
   command to browse locally (`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).
6. **Inference performance** (span-12, two clearly separated halves):
   - `Stored offline benchmark` (slate badge): S-learner latency p50/p95 at batch 1/1k/100k,
     rows/s, train time, peak RSS (full-scale validation benchmark), with machine/threads note.
   - `Live API probe` (teal `Live` badge): button "Run probe" sends 20 sequential single-record
     `/criteo/predict` requests with a median-feature record and 3 batch requests of 1,000 synthetic
     rows; shows browser-measured round-trip p50/p95 ("includes network and JSON; varies by client").
     Nothing is persisted; never mixed into the stored chart.
7. **Limitations & provenance** (span-12): manifest `limitations`, Criteo caveats, evaluation reports
   paths. **Hillstrom** card: "No served model — fixed policy (Men's E-Mail for all)", frozen config
   and evaluation record facts.

---

## C. Technical architecture

### C1. Stack (pin exact versions at scaffold time)
Next.js (latest stable, App Router, TypeScript strict), Tailwind CSS v4, shadcn/ui (Radix),
Recharts, Lucide, `next/font` Geist. Dev: Vitest + Testing Library + jsdom, Playwright, ESLint
(next config), Prettier. **No** state library, data-fetching library, schema library or CSS-in-JS.
Package manager: npm (pnpm is not installed).

### C2. Folder structure (`frontend/`)
```
frontend/
  app/
    layout.tsx                 # html/body, fonts, globals.css, TooltipProvider
    page.tsx                   # redirect("/overview")
    not-found.tsx
    (workspaces)/
      layout.tsx               # AppShell: sidebar + top bar
      overview/page.tsx        (+ loading.tsx, error.tsx per route)
      experiments/page.tsx
      uplift/page.tsx
      decisions/page.tsx
      operations/page.tsx
  components/
    ui/                        # shadcn generated (button, card, tabs, toggle-group, select, slider,
                               #   tooltip, popover, sheet, dialog, table, badge, skeleton, separator,
                               #   accordion, alert, input, textarea, label, sidebar, scroll-area)
    shell/                     # app-sidebar, top-bar, breadcrumbs, dataset-switch, api-status
    data/                      # page-header, section, kpi-card, interval-text, evidence-badge,
                               #   caveat-callout, source-footnote, hash, data-table, stat-list
    charts/                    # chart-frame, chart-tooltip, theme.ts, forest-plot, budget-value-chart,
                               #   arm-outcome-bars, smd-dot-plot, learning-curve, latency-bars,
                               #   cost-quality-scatter, uplift-rank-chart, histogram, report-figure
    states/                    # skeletons, report-unavailable, model-unavailable, api-offline,
                               #   validation-errors, empty-state
  features/
    overview/ experiments/ uplift/ decisions/ operations/
                               # per-workspace sections; *.client.tsx for interactive islands;
                               # select.ts = pure view-model functions (API payload -> chart rows)
  lib/
    api/
      server.ts                # apiGet<T>(path, {revalidate}) for Server Components -> Result<T>
      browser.ts               # apiPost<T>(path, body, signal) for client islands -> Result<T>
      types.ts                 # hand-written TS types for envelopes and payloads (C4)
      errors.ts                # ApiError {status, code, detail}
    format.ts colors.ts datasets.ts url-state.ts
    cohort.ts                  # seeded synthetic cohort + CSV/JSON parsing/validation
  tests/unit/ tests/e2e/
  next.config.ts               # rewrites /api/v1/* -> ${UPLIFTIQ_API_URL}/api/v1/*
  .env.example                 # UPLIFTIQ_API_URL=http://127.0.0.1:8000
```

### C3. Server vs client responsibilities
- **Server Components** (default): page and section components fetch report endpoints, run the pure
  `select.ts` view-model functions and pass small serializable props down. No API URL or raw report
  shapes reach the browser beyond what a view needs.
- **Client islands** (`"use client"`): selectors bound to URL state, all charts (Recharts needs the
  DOM), Decision Studio (cohort generation, upload parsing, POST requests), Model Ops live probe and
  status refresh, sidebar collapse.
- A page renders each section inside its own `<Suspense fallback={<PanelSkeleton/>}>` with an async
  section component, so slow/missing reports stream and fail independently.

### C4. API client and types
- `apiGet` returns `Result<T> = {ok: true, data: T} | {ok: false, error: ApiError}` instead of
  throwing; sections branch to `ReportUnavailable` / `ApiOffline`. Unexpected exceptions go to
  route `error.tsx`.
- Base URL: server side `process.env.UPLIFTIQ_API_URL`; browser uses same-origin `/api/v1/...`
  through the Next rewrite (no CORS dependency, one origin in a public deployment; the backend's CORS
  settings remain valid for direct use).
- Types: hand-written in `lib/api/types.ts`, one per endpoint payload, mirroring `schemas.py` for
  inference (`Prediction`, `BatchResponse`, `TargetRequest`, `TargetItem`, `TargetResponse`, `Health`)
  and the JSON/CSV structures listed in D for report endpoints (`Envelope<T>` generic with
  `dataset`, `source`, `provenance`, `frozen_config_sha256`, `caveats`, `data: T`). OpenAPI
  generation adds little because `Envelope.data` is `Any`. A contract e2e test (F3) checks every
  endpoint against these types at runtime with small hand-written guards.
- Criteo report values arrive as fractions; conversion to pp happens only in `format.ts`.

### C5. Caching and state
- Report endpoints are immutable historical data: `fetch(..., { next: { revalidate: 3600 } })`.
- `/health`, `/models`, `/models/criteo`, `/models/criteo/runs`: `cache: "no-store"`.
- Mutations (`/criteo/predict*`, `/criteo/target`) only from the browser, never cached.
- State: URL query for view-defining selections (`lib/url-state.ts`: tiny `useQueryState(key,
  default, allowed)` wrapper over `useSearchParams` + `router.replace`, scroll: false); `useState` /
  `useReducer` for Decision Studio; no global store. Dataset choice travels in the query string
  between workspaces (sidebar links preserve `?dataset=` when the destination supports it).
- `lib/datasets.ts` declares per-workspace support and defaults:
  overview `[hillstrom, criteo]` default hillstrom; experiments both, default hillstrom;
  uplift both, default criteo; decisions both, default criteo; operations both (Criteo primary,
  Hillstrom card), no switch.

### C6. Charts
- One `charts/theme.ts` (colors, font sizes, margins, grid style, animation flag) consumed by every
  chart; wrappers accept already-shaped rows from `select.ts` and never fetch.
- Dot-and-whisker/forest plots: `ComposedChart` with `Scatter` + `ErrorBar` (vertical layout for
  forest plots). Bands: `Area` with `[low, high]` data. Reference lines: `ReferenceLine` /
  `ReferenceArea`. Log x-axis for learning/training-time charts (`scale="log"`, explicit ticks).
- Large client data (Decision Studio, up to 10,000 rows): pre-bin the histogram, downsample the rank
  chart to ≤ 500 points (keep exact cut-off point), scatter ≤ 2,000 sampled points; table paginated.
- `ChartFrame` provides the "View as table" alternative for every chart (accessibility + auditing).

### C7. Synthetic cohort generator (`lib/cohort.ts`)
- Input: per-feature `{min, q01, q25, q50, q75, q99, max}` from `/criteo/diagnostics`.
- Seeded PRNG (mulberry32) → for each feature independently, sample u ∈ [0,1) and invert the
  piecewise-linear CDF through the knots (0,min),(.01,q01),(.25,q25),(.5,q50),(.75,q75),(.99,q99),(1,max);
  repeated knots reproduce point masses. Ids `0..n-1`. Deterministic for (seed, n).
- Always flagged `synthetic: true` in UI state and exports. Unit-tested for determinism, bounds and
  schema validity.

### C8. Error handling summary
| Backend response | UI |
|:--|:--|
| 404 `report_unavailable` | section-level `ReportUnavailable` with path |
| 404 on `/models/criteo/runs` | MLflow fallback panel (expected in public deployments) |
| 503 `model_unavailable` | `ModelUnavailable`; Decision Studio run disabled; status strip degraded |
| 422 `invalid_request` | field/row errors mapped from `detail[].loc` (`records.<i>.<field>`) |
| 422 `invalid_input` | message banner (e.g. batch over limit) |
| network failure | `ApiOffline` with retry; top-bar status Offline |
| `/health.status = degraded` | amber status, list of unavailable reports |

---

## D. Backend integration map

### D1. Existing endpoints used per workspace (all under `/api/v1`)
| Workspace | Endpoint → fields used |
|:--|:--|
| Shell | `GET /health` → status, criteo_model_loaded, model_version, model_error, reports_available |
| Overview | `GET /overview` → hillstrom.{frozen_policy, value, vs_strongest_alternative_baseline, heldout_customers, caveats}, criteo.{model, heldout_customers, evaluation_record_status, code_dirty, caveats}; `GET /datasets`; `GET /hillstrom/decision` → policy_values; `GET /criteo/heldout` (numeric, see G1) |
| Experiment Lab · Hillstrom | `GET /hillstrom/groups` (group_summary rows), `/hillstrom/effects` → full_sample, heldout (effect rows with CI, p, Holm, significance), `/hillstrom/balance`, `/hillstrom/figures/{spend_distribution, covariate_distributions, …}` |
| Experiment Lab · Criteo | `GET /criteo/diagnostics` → dataset (arms, features, overall), treatment (smd, ks, classifier, checks, verdict), split; `/criteo/ipw-aipw` → estimates, paired_intervals, arm_rates, permutation_test, poststratified_on_predicted_baseline, strata, feature_balance; `/criteo/figures/treatment_balance` |
| Uplift Explorer · Criteo | `GET /criteo/benchmark` → selection (order, rows, steps, chosen, best_by_value, best_by_qini, qini_supports_choice), change_from_2m, learning_curve.levels / changes, tables.* (display), + G2 numeric; `/criteo/heldout`, `/criteo/curves` → figures; `/criteo/figures/{heldout_gain_curve, heldout_value_vs_random, scaling_curves}` |
| Uplift Explorer · Hillstrom | `GET /hillstrom/models` → candidates (selection_summary rows), selection |
| Decision Studio · Criteo | `POST /criteo/target` {records[f0–f11, id?], budget}, `POST /criteo/predict` {f0–f11}; `GET /criteo/heldout` (validated budgets); `GET /criteo/diagnostics` (quantiles for cohort/medians); `GET /models/criteo` (seed, version) |
| Decision Studio · Hillstrom | `GET /hillstrom/decision` → chosen_policy, policy_values, comparisons, evaluation_record, outcome_roles |
| Model Operations | `GET /health`, `/models`, `/models/criteo` → loaded, load_error, manifest, evaluation_record; `/models/criteo/runs` (404-tolerant); `/criteo/benchmark` (stored engineering via G2); live probe `POST /criteo/predict`, `/criteo/predict/batch` |

### D2. Genuine gaps
- **G1 — Criteo held-out numbers are served only as display strings.** `/criteo/heldout`,
  `/criteo/curves` and `/overview.criteo.budgets` return `tables.*` rows like
  `"0.582 [0.540, 0.617]"`. The numeric rows already exist in `reports/criteo/final/final_results.json`
  (`policies`, `references`, `qini`) but are not passed through. Charts with CIs need numbers.
  **Smallest fix (additive, ~10 lines):** add `numeric: {policies, references, qini}` from
  `final_results.json` to `/criteo/heldout` and `/criteo/curves`. No recomputation; existing fields unchanged.
- **G2 — Validation benchmark numeric rows and engineering measurements are not exposed.**
  `/criteo/benchmark` returns display-string tables; the numeric `policies`, `qini`, `pairwise`
  (in `benchmark_full/final_results.json`), per-model `manifests` (fit_seconds, latency list, peak RSS,
  hyperparameters, TARNet training, libraries, code) and `scaling_results.json.engineering`
  (train seconds and RSS at 100K/500K/2M) are not passed through.
  **Smallest fix (additive):** add `numeric: {policies: policies[<any reference>], qini: qini[<any
  reference>], pairwise, models: {name: manifest subset}}` and `learning_curve.engineering`.
- Everything else is covered: synthetic cohorts, medians, uplift ranking chart, budget evidence
  lookup, CSV export and the live latency probe are **frontend-only**.
- Not gaps (accepted limits, designed around): no gain-curve point data anywhere (stored PNGs only);
  held-out estimates only at 10/20/30%; 100K/500K/2M latency not stored (latency shown for full-scale
  finalists only); Hillstrom spend-distribution/covariate tables not served (stored figures used);
  MLflow DB and `model.txt` are git-ignored, so a public deployment returns 404/503 for runs/inference
  unless the package is shipped (fallbacks designed above).

### D3. Supported interactions vs frontend-only presentation
- Backed by the API: targeting at any budget on supplied records; single and batch prediction; health;
  reading all stored results and figures.
- Frontend-only presentation: dataset/outcome/scope/model selectors (filtering already-returned data),
  sorting, "View as table", synthetic cohort generation, upload parsing/validation, downsampling,
  CSV export, per-1,000 rescaling of Hillstrom held-out rates, browser-measured latency probe,
  matching a chosen budget to the stored 10/20/30% evidence.

---

## E. Implementation roadmap

### F1 — Foundation + application shell + Overview
Scope
- Fix-first: apply G1 and G2 (additive backend pass-through + tests) if approved.
- Scaffold `frontend/` (Next.js App Router, TS strict, Tailwind v4, shadcn init, Geist, ESLint,
  Prettier, Vitest), rewrites, `.env.example`, npm scripts (`dev`, `build`, `lint`, `typecheck`, `test`).
- Design tokens (A1–A3), `lib/format.ts`, `lib/colors.ts`, `lib/datasets.ts`, `lib/url-state.ts`,
  API layer (`server.ts`, `browser.ts`, `types.ts`, `errors.ts`).
- Shell: sidebar (expanded/collapsed/sheet), top bar with breadcrumbs, dataset switch, API status.
- Shared components: page-header, section, kpi-card, interval-text, evidence-badge, caveat-callout,
  source-footnote, hash, data-table, all `states/`, `ChartFrame`, `ChartTooltip`, chart theme,
  `ForestPlot` (dot-and-whisker), `BudgetValueChart`, `ReportFigure`.
- Overview page complete for both datasets.
Acceptance criteria
- `npm run build`, `lint`, `typecheck` pass; no runtime console errors.
- Overview shows only API-derived values (matches `/overview` and `/criteo/heldout` exactly after
  formatting); Criteo card shows the "Benchmark, not business ROI" label and caveats.
- Stopping the backend shows `ApiOffline` per section and "Offline" status, without a crash; removing a
  report shows `ReportUnavailable` for that section only.
- Sidebar works by keyboard, collapses, persists state, becomes a sheet below 768 px; no horizontal
  page scroll at 375 px.
Testing
- Vitest: every formatter (units, signs, CI, p-values, hashes, Criteo-never-currency), URL-state
  parsing, `select.ts` for Overview, API `Result` error mapping (404/422/503/network).
Out of scope
- Other workspaces (placeholders with "Coming in F2/F3"), dark mode, deployment.

### F2 — Experiment Lab + Uplift Explorer
Scope
- Experiment Lab (Hillstrom + Criteo variants) and Uplift Explorer (Criteo validation + held-out
  scopes, Hillstrom variant), including `ArmOutcomeBars`, `SmdDotPlot`, `EstimatorComparison`,
  `StrataChart`, `Leaderboard`, `PairwiseForest`, `LearningCurve`, `TrainingTimeChart`,
  `CostQualityScatter`, `LatencyBars`, `SelectionRationale`, `ModelDetailSheet`.
- URL-bound selectors (outcome, sample, scope, compared models, metric).
Acceptance criteria
- Changing outcome/sample/scope/models/metric updates every dependent chart and table and the URL;
  reload restores the view.
- Significance badges are computed from `significant_holm_0.05` and outcome role, not hardcoded.
- IPW/AIPW panel is visible by default on the Criteo Experiment Lab; held-out scope shows only
  the S-learner; no chart draws values between 10/20/30% or a curve not backed by stored points;
  validation and held-out are visually distinct (violet vs teal chips) and never on one chart
  without labels.
- Learning curve shows 6 models through 2M and 3 at full scale exactly as in the reports.
Testing
- Vitest for every `select.ts` (row counts, CI mapping, pp conversion, selection steps) using
  small inline fixtures copied from committed report structure (test-only).
- Component tests for selectors → chart props; "View as table" renders the plotted values.
Out of scope
- Inference, Decision Studio, Model Ops; any new metric not present in reports.

### F3 — Decision Studio + Model Operations + final polish
Scope
- Decision Studio (Criteo live targeting with synthetic/upload/single input; Hillstrom policy board),
  `lib/cohort.ts`, `UpliftRankChart`, `UpliftDistribution`, `ProbabilityScatter`, `RankedTable`,
  `ValidatedEvidence`, `HowThisWasDecided`, CSV export.
- Model Operations (status, model card, reproducibility, lineage, config, MLflow with fallback,
  stored vs live performance, limitations).
- Responsive pass on all pages (A8), accessibility pass (keyboard, focus, contrast, reduced motion),
  loading/error polish.
- Playwright e2e against a locally running backend.
Acceptance criteria
- Targeting results equal the API response (k = ⌊budget·n⌋, ranks 1..n, selected = rank ≤ k);
  same seed + size gives the same cohort and the same ranking.
- Budgets 10/20/30% show the held-out estimates; any other budget shows the "no validated estimate"
  message and no number.
- Synthetic badge is visible wherever synthetic data is shown and in exports.
- With the model package removed: Decision Studio shows `ModelUnavailable`, Model Ops shows degraded
  status and the manifest from `reports/criteo/model_package.json`; with `mlruns/` absent: MLflow
  fallback panel.
- Stored benchmark and live probe are separate panels with different badges.
- Lighthouse accessibility ≥ 95 on each page; usable at 375, 768, 1280, 1440 px.
Testing
- Vitest: cohort determinism/bounds/schema, upload parser errors (missing/extra column, NaN, duplicate
  or partial ids, > 10,000 rows), budget→evidence lookup, 422 `loc` → row/field mapping.
- Playwright: smoke through all five pages; Decision Studio run at 20% and 37%; invalid upload;
  model-unavailable scenario (backend started with a missing package dir); API contract check —
  every GET endpoint's payload satisfies the type guards in `lib/api/types.ts`.
Out of scope
- Auth, persistence of decisions, deployment/hosting, dark mode, Hillstrom per-customer scoring,
  new endpoints beyond G1/G2.

---

## F. Risks and decisions

Decisions needed before coding
1. **Approve G1 + G2** (additive numeric pass-through in three report endpoints, with tests). Without
   it the frontend would parse display strings like `"0.582 [0.540, 0.617]"`, which loses precision
   and breaks if a report's formatting changes. Recommended: approve.
2. **Repair local Node.js.** `/opt/homebrew/bin/node` (26.9.0) fails to start: missing
   `libsimdjson.33.dylib` (Homebrew dependency drift). Fix: `brew reinstall simdjson node` (or
   install Node 22 LTS). Nothing in F1 can run until this works.

High-impact risks
- **Over-claiming through visuals** (implied curves, budget interpolation, mixed datasets, currency).
  Mitigation: chart rules A6, typed formatters, F2/F3 acceptance checks above.
- **Public deployment lacks `model.txt` and `mlruns/`** (both git-ignored): inference returns 503 and
  runs 404. Designed fallbacks cover it; shipping the model artifact is a deployment decision for later.
- **Client performance at 10,000 rows** in Decision Studio: mitigated by downsampling, binning and
  pagination (C6); the API itself measured ~20 ms per 1,000 records locally.
