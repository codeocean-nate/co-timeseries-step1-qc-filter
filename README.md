# Instrument time-series demo — step 1: QC filter

Part of the Code Ocean instrument time-series orchestration demo (`co-timeseries-app`
holds the Streamlit orchestrator and its SETUP.md). Reads the raw readings mounted under
`/data` and writes QC-filtered readings plus the evidence for every decision to
`/results`, which the orchestrator captures as the `clean-readings` data asset for the
analysis step.

- **Environment**: Python 3.9+ with pip packages `pandas`, `numpy`.
- **Input**: any CSV data asset. The capsule scans `/data` recursively, so the mount name
  is not load-bearing; the readings file is picked by name and shape (see
  [Choosing the readings file](#choosing-the-readings-file)). Every other CSV — e.g.
  `instruments.csv` — is copied through to `/results` unchanged so it travels with the
  dataset (see [Pass-through files](#pass-through-files)).
- **Output**: `clean_readings.csv`, `qc_flags.csv`, `qc_summary.csv`, `manifest.json`.
- **Local test**: `DATA_DIR=<dir with readings.csv> RESULTS_DIR=<out dir> bash code/run`

## Outputs

| File | Contents |
|---|---|
| `clean_readings.csv` | the rows that passed, with exactly the input's columns |
| `qc_flags.csv` | every input row plus `qc_status` (`kept`/`dropped`) and `qc_reason` |
| `qc_summary.csv` | per instrument: `n_input`, `n_kept`, `n_dropped`, one `n_<reason>` count per reason, `coverage_pct`, `kept` |
| `manifest.json` | `step`, `parameters` / `effective_parameters` / `parameter_warnings` / `parameters_source`, `rows_in`, `rows_out`, `instruments_kept`, `instruments_dropped`, `readings_file` / `readings_file_chosen_by` / `readings_candidates_not_chosen`, `input_files`, `unreadable_input_files`, `outputs`, `passthrough_files`, `input_files_not_copied`, `spike_rule_skipped_instruments`, `n_unknown_instrument_rows` / `unknown_instrument_label` / `n_rows_with_literal_unknown_instrument_id` / `unknown_instrument_label_collision`, `clean_readings_missing_downstream_columns`, `timestamps_normalized_to_utc` / `n_timestamps_with_utc_offset` / `utc_offsets_seen` / `n_implausible_timestamps` / `plausible_timestamp_range` / `timestamp_interpretation`, `generated_at` (plus reason counts, notes and every warning) |

`qc_reason` is a closed vocabulary of exactly six values: `ok`, `out_of_range`, `spike`,
`flatline`, `low_coverage`, `dropped_instrument`. `manifest.json`'s `reason_counts` always
carries all six keys, on every path — including the runs that found no readings file at
all, where they are all `0`. A reader never has to tell "this reason never fired" from
"this run did not report it".

Two extra counts sit beside them, for the cases where a bare count is what keeps the
manifest from contradicting the data. Both are **subsets** of numbers already reported,
never extra buckets, so `qc_summary.csv`'s row accounting still balances
(`n_input = n_kept + n_dropped`, and the six `n_<reason>` columns sum to `n_input`):

| Key | Meaning |
|---|---|
| `n_unparseable_readings` | rows whose `reading` is not a number. They are counted under `out_of_range` — see [rule 3](#the-qc-rules) |
| `n_unusable_timestamps` | rows whose `timestamp` could not be parsed. They are still QC'd on their reading, but they cannot contribute to `coverage_pct` |
| `n_implausible_timestamps` | rows whose `timestamp` **parses** but lands outside `plausible_timestamp_range`. Same treatment: still QC'd, still written, just excluded from `coverage_pct` — see [Timestamps](#timestamps) |
| `n_unknown_instrument_rows` | rows whose `instrument_id` was **missing** and was relabelled to `unknown_instrument_label` (`(unknown)`) rather than dropped — see [Rows with no instrument id](#rows-with-no-instrument-id) |
| `n_rows_with_literal_unknown_instrument_id` | rows whose `instrument_id` in the **input** already was that literal. Non-zero alongside the row above sets `unknown_instrument_label_collision` — see [Rows with no instrument id](#rows-with-no-instrument-id) |
| `dropped_duplicate_columns` | one record per readings-file column dropped because its name collided with an earlier one once trimmed and lower-cased — see [Column names that collide](#column-names-that-collide). Empty on a normal run |

Four more keys record what parsing the timestamp column changed:
`timestamps_normalized_to_utc`, `n_timestamps_with_utc_offset` and `utc_offsets_seen`
(more than one entry there is the mixed-offset case), plus `plausible_timestamp_range`
so the count above can be reproduced — and `timestamp_interpretation`, which says whether
the column was unambiguous ISO-8601 at all. See [Timestamps](#timestamps).

And six keys exist so that **nothing can go missing quietly**. Each one is also named,
in words, in `warnings` (or in `notes` where the situation is expected rather than risky):

| Key | Meaning |
|---|---|
| `unreadable_input_files` | one record per input CSV that could not be read at all (`file`, `error`, `passed_through`) — see [Inputs that cannot be read](#inputs-that-cannot-be-read) |
| `input_files_not_copied` | one record per input file that did **not** reach `/results`, and why — so `outputs` can never be read as "every input survived" |
| `readings_candidates_not_chosen` | one record per **other** CSV with a `reading` column (`file`, `rows`, `reason`, `is_qc_output`) — the files these verdicts are *not* about. Step 2 records the same list for the same mount — see [Choosing the readings file](#choosing-the-readings-file) |
| `readings_file_chosen_by` | why `readings_file` won: a canonical name, or largest-file-wins |
| `spike_rule_skipped_instruments` | instruments the spike rule could not run on, even though `effective_parameters.spike_mad` is set — see [rule 5](#the-qc-rules) |
| `clean_readings_missing_downstream_columns` | columns the **analysis step** requires that `clean_readings.csv` does not have. Non-empty means step 2 will refuse this run's output — see [Will step 2 be able to read this?](#will-step-2-be-able-to-read-this) |

## Choosing the readings file

`/data` can hold more than one CSV, so the readings file is chosen in this order:

1. it must have a **`reading` column** to be a candidate at all;
2. candidates that already carry **`qc_status`/`qc_reason` are skipped**, with a log line
   saying why — those are a previous QC run's *output*, and re-filtering one would bake its
   stale verdicts into this run's "clean" readings;
3. a **canonical name** wins: `readings.csv`, then `clean_readings.csv`;
4. only then, **largest file wins**.

Size alone would be the wrong rule: `qc_flags.csv` is every input row *plus* two verdict
columns, so it is bigger than the `clean_readings.csv` beside it. If a step-1 result asset
is ever re-mounted into step 1, largest-wins would silently pick the flags file.

If *every* candidate turns out to be a QC output, one is used anyway — an empty pipeline is
a worse answer than a re-filtered one — but its stale `qc_status`/`qc_reason` columns are
dropped first, so this run's verdicts are the only ones in the output, and `manifest.json`
records that under `notes`.

**The files that lost are named too.** Choosing is fine; choosing silently is not. Every
other CSV with a `reading` column goes into `readings_candidates_not_chosen` with its row
count and the reason it lost, and the run log and the manifest say that these verdicts
describe **one** of the readings files that were mounted. Without that, this step could QC
a live 48-row `readings.csv` while step 2 analysed the stale `archive_2019_readings.csv`
beside it, both exiting 0 with nothing anywhere admitting the two capsules were describing
different data. **Step 2 now uses this same ordering** and records the same list, so the
two manifests can be laid side by side. A re-mounted `qc_flags.csv` is recorded as a *note*
rather than a warning — it is the expected shape of a captured step-1 result, not a rival.

CSV discovery is recursive and matches the extension **case-insensitively**: pathlib's
`*.csv` glob is case-sensitive on POSIX whatever the filesystem does, so on the
case-sensitive filesystem Code Ocean actually runs, an `INSTRUMENTS.CSV` used to be
invisible — not in `input_files`, never passed through, and with no warning to say a file
had been skipped.

### Inputs that cannot be read

A zero-byte CSV, a binary file with a `.csv` name, a UTF-16 file, or one the mount will not
let us open is **data, not an error** — the run carries on and still exits 0. What it must
not be is *invisible*. Such a file used to be logged once and then dropped on the floor: it
stayed in `input_files` (so it looked consumed), it never reached `outputs`, and `warnings`
and `notes` were both empty. For `instruments.csv` that meant the lookup table silently
failed to travel with the captured result asset while the manifest positively implied it
had. Worse, if the *readings* file was the unreadable one, the manifest said only "no CSV
with a `reading` column was found" — as though the input had contained no readings.

Now every unreadable file gets:

- a `warnings` entry naming the file and the reason;
- a record in `unreadable_input_files` (`file`, `error`, `passed_through`);
- an attempt to copy the **bytes** into `/results` anyway — this step has no opinion about a
  file it could not parse, and the file is still part of the dataset. `passed_through` says
  whether that worked; when it did not (no read permission, say) the file is also in
  `input_files_not_copied` and absent from `outputs`, so nothing implies it survived.

When no readings file is found *and* some input was unreadable, the "wrote empty outputs"
note says so explicitly — the readings file may well be one of them.

### Column names that collide

Every column lookup here is by **trimmed, lower-cased** name, which is what lets the
capsule accept `Timestamp` as `timestamp`. That also means two *distinct* input columns can
collapse onto one name — `Reading,reading`, `Timestamp,timestamp`, `reading ,reading`,
`Instrument_ID,instrument_id` are all pairs pandas keeps apart and this capsule would then
treat as one.

The **first** column of each name wins, the duplicates are dropped at read time, and the
choice is visible three ways: a run-log line, an entry in `warnings`, and one record per
dropped column under `manifest.json`'s `dropped_duplicate_columns` (`column`, `position`,
`normalized_name`, `kept_column`). Because the drop happens before any rule runs, the
duplicate is absent from `clean_readings.csv` and `qc_flags.csv` too — which is exactly why
it has to be recorded rather than quietly resolved. It used to be resolved silently in
favour of the **last** colliding column, so a stray duplicate `reading` full of rubbish
could take over the entire QC pass with an empty `warnings` list; and the header survived
into `clean_readings.csv`, where it crashed the analysis step outright.

## Pass-through files

Every CSV that is *not* the readings file is copied into `/results` unchanged, under its
**input-relative path**: `/data/a/extra.csv` becomes `/results/a/extra.csv`. Copying by
basename instead would silently destroy data — `/data/a/extra.csv` and `/data/b/extra.csv`
would both land on `/results/extra.csv`, the second overwriting the first, while
`manifest.json` listed both as inputs, listed `extra.csv` **twice** under `outputs`, and
said nothing about the loss. Preserving the directory layout means every entry in `outputs`
is a distinct path and each file is listed exactly once; when two input directories do share
a basename, a `notes` entry says so and points at `outputs` for the exact paths.

`manifest.json` names the copies explicitly under **`passthrough_files`**, the subset of
`outputs` that are verbatim copies of an input. Without it an input CSV that happens to be
called `qc_summary.csv` appears in `input_files` *and* in `outputs`, and no reader — human
or machine — can tell whether `/results/qc_summary.csv` holds that input or this run's own
verdicts. (It holds this run's verdicts.)

Nothing is ever overwritten, and each refusal is recorded in `warnings` rather than only in
the run log:

- an input CSV whose relative path **is one of this step's own outputs** —
  `/data/qc_summary.csv` — is not copied, because copying it would overwrite the verdicts
  this run just wrote. Without the warning the manifest read as though that name were both
  an input and an output of this run; the warning says plainly that the `qc_summary.csv` in
  `/results` is this run's own output, not a copy of the input. The comparison **ignores
  case**: every name this step writes is lower case, so a case-sensitive check let
  `/data/Clean_Readings.csv` through on Linux, where it landed in `/results` beside this
  run's real `clean_readings.csv` — and step 2, which matches that filename
  case-insensitively, could then analyse the *unfiltered* pass-through instead of the QC
  output.
- any other destination that already exists is likewise left alone and reported.
- a copy that **fails** (an unreadable file with no read permission is the realistic case)
  never ends the run. It is reported the same way and listed in `input_files_not_copied`,
  because a file that is in neither the results nor the verdicts must not look like an
  output.

## The QC rules

Applied **per instrument, first match wins**, so every row carries exactly one reason:

1. **`dropped_instrument`** — the id was listed in `--drop_instruments`. Ids are matched
   **ignoring case**, and both halves of that are reported: a value that matched nothing at
   all (`--drop_instruments=RX-999`) raises a `warnings` entry listing the ids that *are* in
   the data, and a value that matched under a different spelling (`rx-101` hitting `RX-101`)
   leaves a `notes` entry naming the id actually dropped. Without them a typo simply dropped
   nothing, in silence — while step 2 already warns for the equivalent mistake in
   `--baseline_instrument`.
2. **`low_coverage`** — the instrument's coverage is below `--min_coverage_pct`, where
   coverage = observed readings ÷ expected hourly slots between that instrument's first
   and last timestamp. The whole instrument goes. Only rows with a **parseable** timestamp
   can count towards coverage, and an instrument left with fewer than two of them reports
   `100.0` — so a run containing timestamps pandas cannot read (a typo, a blank, or a date
   outside its representable range) emits a warning naming the count, and the manifest
   carries the same number as `n_unusable_timestamps`. Silence there let a single usable
   row report perfect coverage, which reads as a healthy instrument.

   The ratio only *means* coverage while the readings really are hourly. Readings that
   repeat a timestamp, or arrive faster than hourly, divide by a smaller number than the row
   count: twelve readings all stamped the same instant span 0 hours, expect **one** slot,
   and used to print `coverage_pct 1200.0` into `qc_summary.csv` with nothing to explain it.
   A value above 100 is an artefact of the hourly grid, not a measurement, so it is
   **clamped to 100.0** — nothing is missing — and a `notes` entry names the instrument, the
   raw figure and the reason, calling out the span-0 case by name.
3. **`out_of_range`** — the reading is outside `[--min_reading, --max_reading]`, **or is
   not a number at all**. Those are two different things sharing one reason, and the
   second half does not depend on either bound: an unreadable value is unusable whether or
   not a range is in force. So this reason can fire on a run whose `effective_parameters`
   show `min_reading` and `max_reading` as `null` — a manifest that just said "no bound is
   applied" while `out_of_range` rows appeared would be contradicting its own data. When
   any reading is unparseable the manifest therefore carries a `notes` entry saying the
   reason covers unreadable readings too (and, when no bound is in force, that this is why
   the rows are there), plus `n_unparseable_readings` with the count. The run log's rule
   summary says the same thing: *"range rule: off (no bounds) — a reading that is not a
   number is still flagged out_of_range"*.
4. **`flatline`** — the row is part of a run of `--flatline_run` or more identical
   consecutive readings (a stuck sensor). The **whole run** is flagged, not just the
   repeats.
5. **`spike`** — robust z = `0.6745·(x − median) / MAD` exceeds `--spike_mad`. The median
   and MAD are computed per instrument **after** the out-of-range rows are removed, so a
   355 °C reading cannot inflate MAD and hide the moderate spikes behind it.

   **This rule can be off for one instrument while the manifest shows it on.** An
   instrument whose in-range readings have a **MAD of 0** — a sensor alternating between
   exactly two values, say — has no scale to divide by, so no robust z exists and no reading
   of its can ever be flagged. A 60 °C spike then sails through as `ok`. That used to happen
   with nothing but a stdout line, while `effective_parameters.spike_mad` still read `6`, so
   the artifact everyone inspects afterwards said the rule had run. A skipped rule must say
   so, and here the granularity is the **instrument**: every such instrument is named in a
   `warnings` entry and listed in `spike_rule_skipped_instruments`.
6. **`ok`** — kept.

**A slow drift is signal, not noise.** Drift moves the median along with it, so its robust
z stays small and drifting readings survive the defaults untouched — finding that trend is
the analysis step's job, not QC's.

### Rows with no instrument id

A row whose `instrument_id` is blank is **relabelled, never dropped** — losing a row is
worse than labelling it. The label is `(unknown)` (`manifest.json` carries it as
`unknown_instrument_label` so nobody has to hardcode it), and **step 2 uses the same literal
for the same rows**, so running either capsule on one raw file reports the same instrument
list.

Relabelling is still a *substitution*, and it changes `qc_summary.csv`, `instruments_kept`
and every downstream grouping — so it is never silent. `n_unknown_instrument_rows` carries
the count and a `warnings` entry says plainly that the id in the outputs is this capsule's
label, not a value from the input file.

**Unless the data uses that literal too.** Nothing stops a file from carrying `(unknown)`
as a real `instrument_id`, and when it does, the relabelled rows and the real ones merge
into one group no output can separate — while the warning above would be asserting
something false. So the collision is detected: `n_rows_with_literal_unknown_instrument_id`
counts the rows that came that way, `unknown_instrument_label_collision` flags the merge,
and the warning says instead that the two groups collided and how many rows each
contributed. When the literal appears with *nothing* relabelled, a `notes` entry says the
opposite plainly: that instrument is the input's own value, not this capsule's label.

### Will step 2 be able to read this?

This step tolerates a readings file with no `timestamp` (or no `instrument_id`) column: it
QCs on the reading alone, warns, and writes every output. But `clean_readings.csv` then
inherits the hole, and the **analysis step requires all three columns** — it refuses such a
file and reports no readings at all, which reads as an empty dataset rather than an
incompatible one.

Refusing to *write* the file would be worse (bad data is data, and a missing output breaks
the pipeline harder than a warning does), so instead the run says loudly, in the log and in
`warnings`, that what it just wrote cannot be analysed by the next step — and
`clean_readings_missing_downstream_columns` lists exactly which columns are missing, so a
machine can check it too.

## Timestamps

Timestamps arrive as text and are never trusted. Two things go wrong with them often
enough to have ended a run with exit 1 and a **completely empty** `/results`, and both are
*data* problems, so neither is fatal any more.

**Mixed UTC offsets.** A real export can carry `2026-06-01T00:00:00+00:00` on one row and
`...+05:30` on the next. Plain `pd.to_datetime(errors="coerce")` cannot pick one dtype for
that column and hands back an **object**-dtype index of per-row datetimes — a
`FutureWarning`, not an error. `errors="coerce"` is no protection: it coerces per-*element*
parse failures and says nothing about the dtype of the *result*. This capsule used to limp
along on that object column, exit 0 with an **empty `warnings` list**, and copy the offsets
verbatim into `clean_readings.csv` — where step 2 could not resample them and died with
`TypeError: Only valid with DatetimeIndex ... got an instance of 'Index'`. The column is
now parsed with `utc=True` (per-row `format="mixed"` where the installed pandas supports
it, feature-probed rather than version-sniffed) and the zone dropped, giving one tz-naive
`datetime64` column whose dtype is checked before any rule reads it.

That matters to the *verdicts*, not just to the plumbing: reading a zone-qualified value as
an **instant** is what makes `coverage_pct` and the flatline ordering compare like with
like. So it is never silent — a `warnings` entry and a run-log line say so, and
`manifest.json` carries `timestamps_normalized_to_utc`, `n_timestamps_with_utc_offset` and
`utc_offsets_seen` (more than one entry there *is* the mixed case). **The values written to
`clean_readings.csv` and `qc_flags.csv` are the input's own text, unaltered** — this step
normalizes how it *reads* timestamps, it does not rewrite them. Step 2 does the identical
parse, and so does batch 1's `make_report.py`.

**Implausible instants.** `1700-01-01T00:00:00` — a sentinel or a typo among ordinary 2026
readings — is inside pandas' representable range, so it parses cleanly. Then
`max − min` for that instrument is a span of ~326 years, and an int64 count of
*nanoseconds* overflows at ~292 years: `coverage_percent` raised `OutOfBoundsDatetime` and
the run exited 1 having written nothing at all. One row out of thirteen was enough. Two
independent fixes, because a filter that only catches the cases we thought of is not the
same as arithmetic that cannot fail:

- the span is now measured in **microseconds** (`span_seconds`), a unit whose int64 range
  (±292,471 years) cannot be overflowed by any pair of instants pandas can represent;
- a timestamp outside `plausible_timestamp_range` (`1900-01-01` … `2200-01-01`, recorded in
  the manifest) is excluded from `coverage_pct` exactly as an unparseable one is, and
  **counted** as `n_implausible_timestamps` with a matching `warnings` entry.

**How the column was read, when that was a choice.** ISO-8601 means one thing; other
shapes do not, and the reading this capsule picks is an interpretation the file never
authorised:

- a **bare number** (`1780000000`) in a numeric column is read as *nanoseconds* since
  1970-01-01, so an epoch value in seconds lands two seconds into 1970 — `coverage_pct` is
  then measured over milliseconds;
- a **`01/02/2026`-style date** has its day/month order inferred **per value**, so
  `01/02/2026` (read month-first) and `13/02/2026` (read day-first) in the same column come
  back in two different calendars. That is not a parse failure — it succeeds.

Neither is fixable here (the file does not say what it meant) and neither is fatal, so both
are *recorded*: `manifest.json`'s `timestamp_interpretation` carries `unambiguous_iso8601`,
the count of each shape, up to three example values and — taken from this run's own parse,
not guessed — what each of those examples was actually read as. A column that is not
unambiguous ISO-8601 also gets a `warnings` entry saying it in words. `null` there means
there was no `timestamp` column to look at.

Excluded from the coverage *arithmetic* — **not dropped**. The row is still QC'd on its
reading, still written to `clean_readings.csv` and `qc_flags.csv` with its original
timestamp text, and still counted in `qc_summary.csv`; the instrument is still reported.
Bad data is data.

## Run parameters (App Panel)

`.codeocean/app-panel.json` is what makes this capsule parameterizable. Code Ocean reads
that committed file and materializes an App Panel form — no UI work required — and the
file is read-only in the capsule IDE, so it can only arrive by `git push`. Because the
panel sets `"named_parameters": true`, each value is appended to `code/run` as a single
command-line argument shaped `--param_name=value` (equals sign, not a space); `code/run`
forwards `"$@"` and `qc_filter.py` parses it with `argparse`. Values chosen this way are
recorded on the computation and frozen onto the captured result asset as `app_parameters`,
which is why the orchestrator app routes the user's choices through parameters instead of
keeping them to itself. Adding a parameter here makes it appear in the orchestrator's GUI
automatically — the app renders the panel it reads back from the capsule.

| Argument | Label | Default | Meaning |
|---|---|---|---|
| `--min_reading` | Minimum valid reading | `-20` | below this = `out_of_range`; blank = no lower bound |
| `--max_reading` | Maximum valid reading | `120` | above this = `out_of_range`; blank = no upper bound |
| `--spike_mad` | Spike threshold (MAD multiples) | `6` | robust z limit; `0` disables the rule |
| `--flatline_run` | Flatline run length | `12` | ≥ N identical consecutive readings = stuck sensor |
| `--min_coverage_pct` | Minimum instrument coverage % | `50` | instruments below this are dropped entirely |
| `--drop_instruments` | Drop instruments | *(blank)* | comma-separated ids to force-drop |

Instrument ids match case-insensitively, and the list is split on commas only — so
`--drop_instruments="RX-102, RX-103"` (one argv token containing a space) works.

### The never-fail rules

- **No parameters ⇒ a sensible QC pass.** The panel may not exist on every deployment, so
  the demo must never depend on it.
- **A bad value never fails the run.** It logs a warning, skips *that one rule*, and exits
  0: `--min_reading=abc` drops the lower bound, `--spike_mad=-1` turns spike detection off,
  `--flatline_run=0` turns the flatline rule off, `--min_coverage_pct=999` (outside 0–100)
  turns the coverage rule off. Every warning is echoed into `manifest.json` under
  `parameter_warnings`.
- **"Bad value" includes the ones `float()` accepts.** `nan`, `inf`, `-inf`, `Infinity`
  and `1e400` (which overflows to inf) all parse fine and then do real damage:
  `int(nan)` raises `ValueError` and `int(inf)` raises `OverflowError`, either of which
  would end the run with an empty `/results`; and a `nan` threshold makes every comparison
  false, so its rule stops firing while the log still says it is on. `parse_float` rejects
  all of them — and absurdly large finite values, which are the same silent death without
  the crash — with a warning naming the value. `--flatline_run` is additionally clamped to
  100000, again with a warning.
- **A skipped rule always says so.** Every `null` in `effective_parameters` has a matching
  entry in `parameter_warnings` (a value that was rejected) or in `notes` — `--spike_mad=0`,
  the panel's documented off-switch, and `--min_reading=` and friends, where a blank
  deliberately turns a rule off. Nothing was rejected in those two cases, so they are not
  warnings, but the resulting `null` still owes the reader an explanation. A rule that goes
  quiet with no explanation anywhere is a contract violation, not merely untidy.
- **`manifest.json` is always valid JSON.** Bare `NaN`/`Infinity` literals are not — strict
  parsers and `JSON.parse` reject them — so the manifest is written with
  `allow_nan=False`. That is a backstop behind the parse-time checks, not the fix.
- **A fractional value for a whole-number parameter warns before it is truncated.**
  `--flatline_run=2.9` is used as `2`, and says so: *"Flatline run length "2.9" is not a
  whole number — the fractional part was discarded and 2 was used."* Every other coercion
  in this capsule emits a message, so a silent one is the one an operator would miss.
- **An unknown parameter is ignored, not fatal.** A stale App Panel that sends
  `--not_a_param=x` logs a warning and the run continues; the token is listed in
  `parameters_source.ignored_tokens` and named in a `parameter_warnings` entry.
- **A token the run could not use never becomes a silent lie.** There are two ways a token
  ends up unused, and both are handled the same way:
  - argparse **rejects** the list — `--spike_mad=99 --min_coverage_pct`, a flag with no
    value, raises;
  - argparse **accepts** the list but does not consume all of it — an unknown parameter, a
    stray word, or anything after an end-of-options `--`.

  Either way the run survives by **partial recovery**: the tokens argparse can actually use
  are re-parsed and honoured (`spike_mad` really is 99), the rest are discarded, and one
  `parameter_warnings` entry quotes the raw argument list and names exactly what was thrown
  away. The second case was the dangerous one, because argparse reported *success*:
  `--  --spike_mad=4` left the operator's 4 with no effect at all while the manifest read
  `spike_mad "6"`, `parameter_warnings []`, `argv_parsed true` **and**
  `parameters_supplied ["spike_mad"]` — a manifest claiming the value was supplied and
  recording the default in the same breath. The invariant now enforced everywhere:
  **`parameters_supplied` never names a parameter whose recorded value is its
  default-because-we-could-not-use-it.** `parameters_source.argv` puts the raw list on the
  record either way, so no reader can mistake a fallback for a choice, and the run log
  matches: it reports the parameters that were **honoured**.
- **A filter that removes everything is an answer, not an error.** `clean_readings.csv` is
  still written *with its header*, `qc_summary.csv` shows `kept=False` and the reason
  counts that explain it, `manifest.json` carries a note, and the run exits 0. An input
  that was **already empty** gets a different note — *"the readings file … has no data
  rows — QC removed nothing because there was nothing to remove"* — because telling a
  reader QC dropped rows that never existed is the same defect in miniature.
- **No readings file at all** produces the same four outputs, empty but valid, and exits 0,
  with `reason_counts` carrying its usual six zeroed keys.

### The manifest's parameter block

Every capsule in this demo records parameters the same three ways, so one reader handles
all of them:

| Key | Contents |
|---|---|
| `parameters` | the raw values as received *and understood* (strings), including untouched defaults |
| `effective_parameters` | the coerced values the run actually used, with `null` for a rule that was skipped |
| `parameter_warnings` | one message per rejected, clamped or truncated value, the same text as the run log |
| `parameters_source` | how those values arrived: `argv` (the raw argument list, verbatim), `argv_parsed` (`true` only when the list was used **exactly as sent** — `false` if argparse rejected it *or* left any token unconsumed), `parameters_supplied` (the names actually honoured), `ignored_tokens`, `superseded_tokens` |

So `--spike_mad=xyz` shows up as `"spike_mad": "xyz"` in `parameters`, `null` in
`effective_parameters` (the spike rule was skipped), and one entry in `parameter_warnings`
— and the run still exits 0. `warnings` keeps the full list, parameter and data alike.

`superseded_tokens` covers the one way a value can be discarded by a list that parsed
*perfectly*. `--spike_mad=1 --spike_mad=99` is a clean parse: argparse takes the last value
and never mentions the first, so the manifest read `spike_mad "99"`, `argv_parsed true` and
— the false part — `ignored_tokens []`, positively claiming nothing had been dropped. Every
overridden token is now listed there and named in a `parameter_warnings` entry that says
which value won.

`parameters_source` exists because `parameters` **cannot** express "you sent
`--spike_mad=99` and I could not use it" — in that slot it holds the *default*, which a
reader would otherwise take for the operator's choice. For a capsule whose whole purpose is
provenance, a manifest that quietly contradicts its caller is worse than a crash: a crash
is loud, and this is the artifact people trust afterwards. So the raw argument list is
always on the record, parsed or not.

Local test with parameters:

```bash
DATA_DIR=<dir with readings.csv> RESULTS_DIR=<out dir> bash code/run \
  --min_reading="0" --max_reading="50" --spike_mad="3" \
  --flatline_run="5" --min_coverage_pct="90" --drop_instruments="RX-103"
```
