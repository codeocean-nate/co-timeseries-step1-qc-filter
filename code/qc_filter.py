"""Time-series step 1 capsule: QC-filter raw instrument readings.

Code Ocean concepts demonstrated here:

* Input data assets are mounted READ-ONLY under ``/data/<mount>/...``. This
  capsule never assumes a mount name — it scans ``/data`` recursively for CSVs
  (extension matched ignoring case, because pathlib's glob is case-sensitive
  on POSIX and Code Ocean's filesystem is too) and picks the readings file by
  name and shape, naming every candidate it did NOT pick (see
  ``pick_readings_file``), so the demo works whatever the asset happens to be
  mounted as and step 2 can be seen to have read the same file.
* Everything written under ``/results/`` is captured when the computation
  finishes, which is how the next step gets its input: the orchestrator app
  turns this run's results into the ``clean-readings`` data asset.
* Capsules must not assume network access at runtime — this one is pure
  pandas/numpy on the mounted files.
* Progress lines printed to stdout show up in the Code Ocean run log.

Local-test override: ``DATA_DIR`` replaces ``/data`` and ``RESULTS_DIR``
replaces ``/results``.

Run parameters (the human-in-the-loop part of the demo):

* This capsule has NO App Panel and does not need one. **Never add a
  ``.codeocean/app-panel.json``**: that file does not create a panel, and its
  presence makes every run of the capsule fail with ``403 corrupted object
  files``. The orchestrator app declares these parameters itself and sends them
  as *named run parameters*, which Code Ocean appends to ``code/run`` as
  command-line ARGUMENTS — not env vars. Each value arrives as one argv token
  shaped ``--param_name=value`` (equals sign, not a space), which ``argparse``
  parses natively.
* Routing the GUI's choices through Code Ocean parameters (instead of keeping
  them inside the calling app) is what makes them show up on the computation
  and frozen onto the captured result asset as ``app_parameters`` — i.e. it is
  what makes the QC thresholds part of the recorded provenance.
* Every parameter is OPTIONAL with a working default, and a bad value never
  fails the run: it logs a warning, skips that ONE rule, and carries on. A
  filter that would remove every row still writes valid outputs (an empty
  ``clean_readings.csv`` *with* its header) and exits 0 — the demo must never
  dead-end on a typo. "Bad value" includes ``nan``, ``inf``, ``-inf``,
  ``Infinity`` and absurdly large numbers, all of which ``float()`` accepts
  without complaint; see ``parse_float`` for why each one is poison.
* A rule that gets skipped SAYS SO, in the run log and in the manifest's
  ``parameter_warnings``. Quietly comparing readings against ``nan`` would
  turn a rule off while the log still claimed it was on.
* The manifest must never CONTRADICT the caller. This capsule exists to
  demonstrate provenance, so a manifest that reads "you sent spike_mad=6" when
  the operator sent 99 is the worst defect available here — silent, and baked
  into the artifact everyone trusts afterwards. Two rules follow from that:
  ``manifest.json`` always carries the raw argument list verbatim under
  ``parameters_source``, and anything the run could not use produces a
  ``parameter_warnings`` entry rather than a quiet fallback (see
  ``parse_parameters``). The invariant both rules serve:
  ``parameters_supplied`` must never name a parameter whose recorded value is
  its default-because-we-could-not-use-it.
* Column lookups are by trimmed, lower-cased name, so two distinct input
  columns can collapse onto one (``Reading,reading``). The FIRST of each name
  wins, the duplicates are dropped at read time, and that choice is logged,
  warned about and listed in the manifest — it used to be resolved silently in
  favour of the LAST one, and it crashed the analysis step outright. See
  ``drop_duplicate_columns``.
* Timestamps are the other column that could end a run with nothing written,
  and for two unrelated reasons: a column of MIXED UTC offsets has no single
  dtype (``pd.to_datetime`` hands back an OBJECT index, which this step limped
  along on and step 2 could not resample at all), and a single implausibly old
  instant makes ``max - min`` overflow int64 nanoseconds. Both are DATA, so
  neither is fatal: the column is parsed to one tz-naive ``datetime64`` dtype
  and the span is computed in a unit that cannot overflow. Nothing is dropped
  for it — such rows are still QC'd, still written with their ORIGINAL text,
  and still counted; they are only kept out of coverage_pct. See
  ``parse_timestamp_series``, ``span_seconds``, and the manifest's
  ``timestamps_normalized_to_utc`` / ``n_implausible_timestamps``.

QC rules, applied per instrument, first match wins so every row carries
exactly one reason:

    1. dropped_instrument  the id was listed in --drop_instruments
    2. low_coverage        the instrument's coverage % is below the threshold
    3. out_of_range        reading outside [min_reading, max_reading], OR not
                           a number at all. The second half does not depend on
                           either bound, so this reason can fire on a run whose
                           effective min_reading/max_reading are both null;
                           when it does, the manifest carries a note saying so
                           and `n_unparseable_readings` gives the count
    4. flatline            part of a run of >= flatline_run identical
                           consecutive readings (stuck sensor)
    5. spike               robust z = 0.6745*(x-median)/MAD exceeds spike_mad,
                           median/MAD computed per instrument AFTER dropping
                           the out-of-range rows so the outliers cannot
                           inflate MAD and hide themselves
    6. ok                  kept

A slow drift is *not* a defect: it moves the median with it, so its robust z
stays small and the drifting rows survive the default parameters. That is
deliberate — the analysis step is supposed to find the trend.

Outputs to /results:
    clean_readings.csv  rows that passed, same columns as the input
    qc_flags.csv        every input row + qc_status + qc_reason
    qc_summary.csv      one row per instrument: counts, coverage, kept
    manifest.json       provenance for the analysis step
Any other CSV found in the input (e.g. ``instruments.csv``) is copied through
unchanged so downstream steps still see it, under its INPUT-RELATIVE path
(``/data/a/extra.csv`` -> ``/results/a/extra.csv``) so that two files sharing a
basename cannot overwrite each other; see ``passthrough``.
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Locations (Code Ocean conventions, overridable for local testing)
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/results"))

# Column names in the demo readings schema. Only `reading` is load-bearing
# (it identifies the readings file); the others degrade gracefully.
READING_COL = "reading"
TIMESTAMP_COL = "timestamp"
INSTRUMENT_COL = "instrument_id"

# The label a row with a MISSING instrument_id is filed under. Rows are
# relabelled rather than dropped (losing a row is worse than labelling it), and
# step 2 uses the same literal for the same rows, so running either capsule on
# the same raw file reports the same instrument list. The count is reported in
# the manifest as `n_unknown_instrument_rows` and named in `warnings`: a
# renamed id is still an altered value, and altering data in silence is the one
# thing these capsules promise never to do.
UNKNOWN_INSTRUMENT = "(unknown)"

# Values that mean "this row has no instrument_id". `astype(str)` turns a
# genuinely missing cell into the literal "nan", so the blank test has to run
# against the stringified form and cover that spelling too — which is exactly
# the guard step 2 was missing.
MISSING_INSTRUMENT_TOKENS = ["", "nan"]

# Header used for the empty outputs when the input has no readings file at
# all, so downstream steps still get parseable CSVs instead of nothing.
FALLBACK_COLUMNS = ["timestamp", "instrument_id", "metric", "reading", "operator"]

# Readings filenames this pipeline knows, most-preferred first. Choosing by
# NAME before size is what stops a re-mounted QC result asset from hijacking
# the run (see pick_readings_file).
CANONICAL_READINGS_NAMES = ["readings.csv", "clean_readings.csv"]

# Columns that identify a CSV as some previous QC run's OUTPUT rather than raw
# readings. Filtering such a file would carry its stale verdicts into this
# run's "clean" output.
QC_OUTPUT_COLUMNS = ["qc_status", "qc_reason"]

# The closed vocabulary of qc_reason values. qc_summary gets one count column
# per reason, named n_<reason>.
REASONS = [
    "ok",
    "out_of_range",
    "spike",
    "flatline",
    "low_coverage",
    "dropped_instrument",
]
REASON_COUNT_COLS = ["n_" + r for r in REASONS]
SUMMARY_COLUMNS = (
    [INSTRUMENT_COL, "n_input", "n_kept", "n_dropped"]
    + REASON_COUNT_COLS
    + ["coverage_pct", "kept"]
)

# The three CSVs this step always writes, in manifest order, plus the manifest
# itself. One list so the `outputs` key and the pass-through collision check
# (see passthrough) can never drift apart: an input CSV that happens to be
# called qc_summary.csv must not be copied over this run's own verdicts.
DELIVERABLE_CSVS = ["clean_readings.csv", "qc_flags.csv", "qc_summary.csv"]
OWN_OUTPUTS = frozenset(DELIVERABLE_CSVS + ["manifest.json"])
# ...and the same names lower-cased, because the collision check has to ignore
# case. Every name above already is lower case, so a case-SENSITIVE comparison
# let `Clean_Readings.csv` through on a case-sensitive filesystem — where it
# then sat in /results alongside this run's real clean_readings.csv, and step 2
# (which matches that filename ignoring case) could pick the unfiltered copy.
OWN_OUTPUTS_LOWER = frozenset(n.lower() for n in OWN_OUTPUTS)

# Robust-z scale factor: 0.6745 = the 75th percentile of a standard normal,
# which makes MAD-based z comparable to an ordinary standard-deviation z.
MAD_SCALE = 0.6745

# Sanity bounds for the numeric parameters. ``float()`` is far more permissive
# than it looks: it accepts "nan", "inf", "-inf" and "Infinity", and quietly
# overflows "1e400" to inf. Every one of those poisons the run downstream (see
# parse_float), and so does a huge-but-finite threshold, so both are rejected
# where the value is parsed rather than where it eventually explodes.
MAX_ABS_THRESHOLD = 1e9     # no real reading or MAD multiple is this large
MAX_FLATLINE_RUN = 100000   # no real sensor repeats one value this many times

# argparse's end-of-options marker. Everything after one is forced to be a
# POSITIONAL argument, and this capsule defines no positionals — so argparse
# reports a successful parse while quietly handing back "--spike_mad=4" as
# unrecognized. See resolve_argv for why that is the worst possible outcome
# here and what happens instead.
END_OF_OPTIONS = "--"


def log(msg):
    # type: (str) -> None
    """Print a progress line (shows up in the Code Ocean run log)."""
    print("[qc] {}".format(msg), flush=True)


# ---------------------------------------------------------------------------
# Run parameters (Code Ocean named run parameters)
# ---------------------------------------------------------------------------
# This table is the source of truth for what the capsule ACCEPTS. There is no
# App Panel to keep in sync — and there must never be one, see the module
# docstring. The orchestrator app mirrors these names and defaults in its own
# form.
#   (param_name / argument key, label shown on the panel, default)
PARAM_SPECS = [
    ("min_reading", "Minimum valid reading", "-20"),
    ("max_reading", "Maximum valid reading", "120"),
    ("spike_mad", "Spike threshold (MAD multiples)", "6"),
    ("flatline_run", "Flatline run length", "12"),
    ("min_coverage_pct", "Minimum instrument coverage %", "50"),
    ("drop_instruments", "Drop instruments", ""),
]
PARAM_LABELS = dict((name, label) for name, label, _ in PARAM_SPECS)
PARAM_DEFAULTS = dict((name, default) for name, _, default in PARAM_SPECS)


def build_parser():
    # type: () -> argparse.ArgumentParser
    """The argparse parser for the App Panel's arguments."""
    parser = argparse.ArgumentParser(
        add_help=False,      # a stray -h must not short-circuit the QC run
        allow_abbrev=False,  # only exact --param_name keys, no fuzzy matching
    )
    for param_name, label, default in PARAM_SPECS:
        parser.add_argument("--" + param_name, default=default, help=label)
    return parser


def format_argv(argv):
    # type: (List[str]) -> str
    """Render an argument list for a log line, unambiguously.

    Tokens that are empty or contain spaces are quoted, so a reader can tell
    ``--drop_instruments="RX-1, RX-2"`` (one token) from two tokens.
    """
    if not argv:
        return "(empty)"
    return " ".join(
        '"{}"'.format(token) if (token == "" or " " in token) else token
        for token in argv)


def split_recoverable_tokens(argv):
    # type: (List[str]) -> Tuple[List[str], List[str]]
    """Split a malformed argument list into re-parseable tokens and the rest.

    Only reached after the whole list has already failed to parse. Because the
    App Panel sends every value as ONE token shaped ``--param_name=value``, a
    token of exactly that shape cannot be the thing argparse choked on — so
    re-parsing just those recovers every well-formed value the operator sent.
    Everything else (a flag with no value, a bare word, a ``--name value`` pair
    whose two halves can no longer be told apart from a stray positional) is
    handed back as un-parseable and reported by name.
    """
    keep = []  # type: List[str]
    dropped = []  # type: List[str]
    for token in argv:
        key = token[2:].split("=", 1)[0] if token.startswith("--") else None
        if "=" in token and key in PARAM_DEFAULTS:
            keep.append(token)
        else:
            dropped.append(token)
    return keep, dropped


def malformed_argv_warning(argv, honoured, ignored):
    # type: (List[str], List[str], List[str]) -> str
    """The parameter_warnings entry for an argument list that was not usable.

    It quotes the RAW argument list, because the whole point is that the
    manifest must not be readable as "these are the values you sent".
    """
    recovered = sorted(supplied_param_names(honoured))
    if recovered:
        return (
            "the argument list could not be used exactly as given ({}) — the "
            "well-formed --name=value token(s) were still honoured ({}), but "
            "these token(s) were discarded and had NO effect on this run: "
            "{}".format(format_argv(argv), ", ".join(recovered),
                        format_argv(ignored)))
    return (
        "the argument list could not be used as given ({}) and no parameter "
        "value could be recovered from it — EVERY supplied value was ignored "
        "and every parameter fell back to its App Panel default; the discarded "
        "token(s) were: {}".format(format_argv(argv), format_argv(ignored)))


def remove_tokens(tokens, unwanted):
    # type: (List[str], List[str]) -> List[str]
    """``tokens`` minus ONE occurrence of each entry in ``unwanted``."""
    remaining = list(unwanted)
    kept = []  # type: List[str]
    for token in tokens:
        if token in remaining:
            remaining.remove(token)
        else:
            kept.append(token)
    return kept


def parse_dropping_unconsumed(parser, tokens):
    # type: (argparse.ArgumentParser, List[str]) -> Tuple[Optional[argparse.Namespace], List[str], List[str]]
    """Parse ``tokens``, dropping whatever argparse refuses to consume.

    ``parse_known_args`` hands back the tokens it could not place, and those
    tokens had NO effect on the namespace it returned — so they are removed and
    the remainder is re-parsed until argparse consumes everything it is given.
    Each pass removes at least one token, so the loop always terminates.

    Returns (namespace, tokens used, tokens dropped), or ``(None, [], tokens)``
    when argparse rejected the list outright (a flag with no value, which
    raises ``SystemExit``).
    """
    used = list(tokens)
    dropped = []  # type: List[str]
    while True:
        try:
            args, unconsumed = parser.parse_known_args(used)
        except SystemExit:
            return None, [], list(tokens)
        if not unconsumed:
            return args, used, dropped
        dropped.extend(unconsumed)
        used = remove_tokens(used, unconsumed)


def resolve_argv(parser, argv):
    # type: (argparse.ArgumentParser, List[str]) -> Tuple[argparse.Namespace, List[str], List[str], bool]
    """Work out which argv tokens this run can actually HONOUR.

    Returns (namespace, honoured tokens, ignored tokens, used_as_given).

    ``used_as_given`` is true only when argparse consumed the whole list
    exactly as it arrived; that is what the manifest reports as
    ``argv_parsed``. Anything less is false — a rejected list, an unknown
    parameter, or a value stranded behind an end-of-options marker.

    That last one is the subtle case and the reason this function exists.
    ``["--", "--spike_mad=4"]`` does NOT raise: argparse accepts it, demotes
    everything after ``--`` to a positional, finds no positionals to fill and
    returns the token as "unrecognized". Treating that as a clean parse
    produced precisely the manifest this capsule must never write — ``spike_mad
    "6"`` (the default), ``parameter_warnings []``, ``argv_parsed true``, and
    ``parameters_supplied ["spike_mad"]`` claiming the operator's 4 was used.
    So the markers are stripped up front and the values behind them are parsed
    like any other, while the markers themselves are reported as ignored.
    """
    ignored = [t for t in argv if t == END_OF_OPTIONS]
    args, honoured, dropped = parse_dropping_unconsumed(
        parser, [t for t in argv if t != END_OF_OPTIONS])
    if args is not None:
        ignored = ignored + dropped
        return args, honoured, ignored, not ignored

    # argparse rejected the list outright. Keep only the tokens that cannot be
    # what it choked on: one self-contained --known=value token each.
    keep, junk = split_recoverable_tokens(argv)
    args, honoured, dropped = parse_dropping_unconsumed(parser, keep)
    if args is None:
        # Unreachable in principle (every kept token is --known=value), but a
        # capsule whose job is provenance does not gamble on "should".
        return argparse.Namespace(**PARAM_DEFAULTS), [], list(argv), False
    return args, honoured, list(junk) + list(dropped), False


def parse_parameters(argv, param_warnings):
    # type: (List[str], List[str]) -> Tuple[Dict[str, str], set, Dict[str, Any]]
    """Parse the App Panel values Code Ocean appended to ``code/run``.

    Named parameters arrive as single argv tokens (``--drop_instruments=RX-102,RX-103``),
    which argparse understands out of the box — a hand-rolled ``--name value``
    parser would silently see nothing.

    Anything argparse cannot USE must not fail the run, and it must not quietly
    become "the operator sent the defaults" either. There are two ways a token
    ends up unused and they used to be handled very differently:

    * a MALFORMED list (``--spike_mad=99 --min_coverage_pct`` — a flag with no
      value) makes argparse raise ``SystemExit``;
    * an ACCEPTED list can still leave tokens unconsumed — a parameter this
      code does not know yet, a stray word, or anything after an end-of-options
      ``--``.

    The second kind was the dangerous one, because argparse reported success:
    the value was dropped, the manifest recorded the DEFAULT in its place with
    an empty ``parameter_warnings``, and ``parameters_supplied`` still named the
    parameter — a manifest that simultaneously claims the value was supplied and
    records the default. Both kinds now go down the same road (``resolve_argv``):

    1. the tokens argparse can actually use are re-parsed and honoured, so the
       operator's real intent survives one bad token elsewhere in the list;
    2. one ``parameter_warnings`` entry quotes the raw argument list and names
       what was discarded;
    3. ``ignored_tokens`` lists the discarded tokens and ``argv_parsed`` goes
       false, so the manifest says the list was not used exactly as sent;
    4. ``parameters_supplied`` is derived from the HONOURED tokens only.

    That last point is the invariant: ``parameters_supplied`` must never name a
    parameter whose recorded value is its default-because-we-could-not-use-it.

    Returns (raw string values, names actually honoured, source dict).
    """
    parser = build_parser()
    args, honoured, ignored, used_as_given = resolve_argv(parser, argv)
    superseded = find_superseded_tokens(honoured)
    source = {
        "argv": list(argv),
        "argv_parsed": bool(used_as_given),
        "parameters_supplied": [],
        "ignored_tokens": list(ignored),
        # Tokens argparse DID understand but a later token of the same name
        # overrode. They are not `ignored_tokens` (the list parsed fine) and
        # they are not honoured either, so without their own key the manifest
        # claimed nothing had been dropped.
        "superseded_tokens": list(superseded),
    }  # type: Dict[str, Any]
    if not used_as_given:
        param_warnings.append(malformed_argv_warning(argv, honoured, ignored))
        log("warning: {}".format(param_warnings[-1]))
    if superseded:
        param_warnings.append(superseded_tokens_warning(superseded, args))
        log("warning: {}".format(param_warnings[-1]))

    out = {}
    for param_name, _label, _default in PARAM_SPECS:
        value = getattr(args, param_name, None)
        out[param_name] = "" if value is None else str(value)
    # Derived from the tokens that were actually USED, not from the raw list:
    # a discarded token must never be recorded as a parameter that was supplied.
    supplied = supplied_param_names(honoured)
    source["parameters_supplied"] = sorted(supplied)
    return out, supplied, source


def supplied_param_names(argv):
    # type: (List[str]) -> set
    """Which parameters were actually passed, vs. left at their default."""
    known = set(PARAM_DEFAULTS)
    supplied = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0]
        if key in known:
            supplied.add(key)
    return supplied


def find_superseded_tokens(honoured):
    # type: (List[str]) -> List[str]
    """Tokens for a parameter that a LATER token of the same name overrode.

    ``--spike_mad=1 --spike_mad=99`` parses perfectly: argparse takes the last
    value and never mentions the first. That left the manifest saying
    ``spike_mad "99"``, ``argv_parsed true`` and — the false part —
    ``ignored_tokens []``, which positively claims nothing was dropped when a
    value the operator sent had in fact been discarded. Every superseded token
    is returned here so the caller can name it in ``parameter_warnings`` and
    record it in ``parameters_source``.
    """
    last_index = {}  # type: Dict[str, int]
    for index, token in enumerate(honoured):
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0]
        if key in PARAM_DEFAULTS:
            last_index[key] = index
    superseded = []  # type: List[str]
    for index, token in enumerate(honoured):
        if not token.startswith("--"):
            continue
        key = token[2:].split("=", 1)[0]
        if key in PARAM_DEFAULTS and last_index.get(key) != index:
            superseded.append(token)
    return superseded


def superseded_tokens_warning(superseded, args):
    # type: (List[str], argparse.Namespace) -> str
    """The parameter_warnings entry for values a later token overrode."""
    names = sorted(set(t[2:].split("=", 1)[0] for t in superseded))
    winners = ", ".join(
        "--{}={}".format(name, getattr(args, name, "")) for name in names)
    return (
        "{} parameter(s) were given more than once ({}); the LAST value of "
        "each won ({}) and the earlier token(s) had NO effect on this run: "
        "{}".format(len(names), ", ".join(names), winners,
                    format_argv(superseded)))


def parse_float(raw, label, warnings, limit=None):
    # type: (str, str, List[str], Optional[float]) -> Optional[float]
    """Parse a numeric parameter. Blank or unusable -> None (rule skipped).

    "Unusable" is deliberately wider than "``float()`` raised", because
    ``float()`` accepts a lot of values this capsule cannot use:

    * ``nan`` — every comparison against nan is False, so the rule it belongs
      to stops firing while the log still claims it is on. A rule that dies
      without a warning is a contract violation, not merely untidy.
    * ``inf`` / ``-inf`` / ``Infinity`` / ``1e400`` (which overflows to inf) —
      same silent death, and ``int(inf)`` raises ``OverflowError`` while
      ``int(nan)`` raises ``ValueError``, either of which would abort the run
      *after* nothing had been written.
    * a huge but finite value, e.g. ``1e40`` — finite, so ``numpy.isfinite``
      is happy, yet still a threshold no reading can ever cross.

    Non-finite values must also never reach ``manifest.json``: ``json`` writes
    them as bare ``NaN``/``Infinity`` literals, which strict JSON parsers
    (including ``JSON.parse``) reject.

    Every rejection appends a warning naming the value and returns None, which
    the caller reads as "turn that one rule off"; the run still exits 0.
    """
    text = (raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        warnings.append(
            "{} \"{}\" is not a number — that rule was skipped".format(label, text))
        return None
    if not np.isfinite(value):
        warnings.append(
            "{} \"{}\" is not a finite number — that rule was skipped".format(
                label, text))
        return None
    if limit is not None and abs(value) > limit:
        warnings.append(
            "{} \"{}\" is larger than the largest sensible value ({:g}) — "
            "that rule was skipped".format(label, text, limit))
        return None
    return value


def note_blank(raw, label, consequence, notes):
    # type: (str, str, str, List[str]) -> None
    """Record that a rule is off because its value was left BLANK.

    Nothing was rejected, so this is not a warning — but it still produces a
    ``null`` in ``effective_parameters``, and every null owes the reader an
    explanation somewhere in the manifest. Same reasoning as ``--spike_mad=0``.
    """
    if not (raw or "").strip():
        notes.append("{} was left blank — {}".format(label, consequence))
        log(notes[-1])


def resolve_settings(params, warnings, notes):
    # type: (Dict[str, str], List[str], List[str]) -> Dict[str, Any]
    """Turn the raw panel strings into typed settings, never raising.

    Any value that cannot be used disables exactly one rule (``None`` means
    "rule off") and appends a warning. The run always continues.

    Every ``None`` in the result must be explainable from the manifest alone,
    so that a reader can tell "the operator turned this off" from "the value
    was garbage": a REJECTED value appends to ``warnings`` (the manifest's
    ``parameter_warnings``), while the one documented off-switch — ``spike_mad``
    of 0 — appends to ``notes`` instead. What must never happen is a null with
    no explanation anywhere.

    Nothing here may raise, and nothing here may leave a non-finite float in
    ``settings`` — those values flow straight into ``manifest.json``, where a
    bare ``NaN`` would make the file invalid JSON. ``parse_float`` enforces
    both, so every number below is already finite and sanely bounded.
    """
    settings = {}  # type: Dict[str, Any]

    # --- range rule --------------------------------------------------------
    note_blank(params["min_reading"], PARAM_LABELS["min_reading"],
               "no lower bound is applied", notes)
    note_blank(params["max_reading"], PARAM_LABELS["max_reading"],
               "no upper bound is applied", notes)
    low = parse_float(params["min_reading"], PARAM_LABELS["min_reading"], warnings,
                      limit=MAX_ABS_THRESHOLD)
    high = parse_float(params["max_reading"], PARAM_LABELS["max_reading"], warnings,
                       limit=MAX_ABS_THRESHOLD)
    if low is not None and high is not None and low > high:
        warnings.append(
            "{} ({}) is above {} ({}) — the range rule was skipped".format(
                PARAM_LABELS["min_reading"], low, PARAM_LABELS["max_reading"], high))
        low, high = None, None
    settings["min_reading"] = low
    settings["max_reading"] = high

    # --- spike rule (0 disables it, per the panel description) -------------
    note_blank(params["spike_mad"], PARAM_LABELS["spike_mad"],
               "spike detection is off", notes)
    spike = parse_float(params["spike_mad"], PARAM_LABELS["spike_mad"], warnings,
                        limit=MAX_ABS_THRESHOLD)
    if spike is not None and spike < 0:
        warnings.append(
            "{} \"{}\" is negative — the spike rule was skipped".format(
                PARAM_LABELS["spike_mad"], params["spike_mad"].strip()))
        spike = None
    if spike == 0:
        # Not a warning: 0 is the panel's documented off-switch, so nothing was
        # rejected. It still goes in `notes`, because effective spike_mad will
        # be null and every null owes the reader an explanation.
        notes.append("{} is 0 — spike detection disabled on purpose".format(
            PARAM_LABELS["spike_mad"]))
        log(notes[-1])
        spike = None
    settings["spike_mad"] = spike

    # --- flatline rule -----------------------------------------------------
    # int(flat) is only safe because parse_float has already rejected nan/inf
    # (int() raises on both) and anything absurdly large.
    note_blank(params["flatline_run"], PARAM_LABELS["flatline_run"],
               "the flatline rule is off", notes)
    flat = parse_float(params["flatline_run"], PARAM_LABELS["flatline_run"], warnings,
                       limit=MAX_ABS_THRESHOLD)
    flat_run = None  # type: Optional[int]
    if flat is not None:
        flat_run = int(flat)   # truncates towards zero: 2.9 -> 2
        truncated = flat_run != flat
        if flat_run < 2:
            warnings.append(
                "{} \"{}\" is below 2 (a run needs at least two readings) — "
                "the flatline rule was skipped".format(
                    PARAM_LABELS["flatline_run"], params["flatline_run"].strip()))
            flat_run = None
        elif flat_run > MAX_FLATLINE_RUN:
            # Clamp rather than skip: the user clearly wanted "only absurdly
            # long runs", and the clamp says so instead of leaving a rule that
            # can never fire.
            warnings.append(
                "{} \"{}\" is above the maximum {} — clamped to {}".format(
                    PARAM_LABELS["flatline_run"], params["flatline_run"].strip(),
                    MAX_FLATLINE_RUN, MAX_FLATLINE_RUN))
            flat_run = MAX_FLATLINE_RUN
        elif truncated:
            # A run length is a count of readings, so 2.9 cannot be obeyed
            # literally. Truncating it silently is the one coercion in this
            # capsule an operator could miss — every other one warns, so this
            # one does too, naming both the value sent and the value used.
            warnings.append(
                "{} \"{}\" is not a whole number — the fractional part was "
                "discarded and {} was used".format(
                    PARAM_LABELS["flatline_run"], params["flatline_run"].strip(),
                    flat_run))
    settings["flatline_run"] = flat_run

    # --- coverage rule -----------------------------------------------------
    # No `limit=` here: a percentage already has a tighter bound of its own
    # just below, and that check gives the better message.
    note_blank(params["min_coverage_pct"], PARAM_LABELS["min_coverage_pct"],
               "the coverage rule is off", notes)
    cov = parse_float(
        params["min_coverage_pct"], PARAM_LABELS["min_coverage_pct"], warnings)
    if cov is not None and (cov < 0 or cov > 100):
        warnings.append(
            "{} \"{}\" is outside 0–100 — the coverage rule was skipped".format(
                PARAM_LABELS["min_coverage_pct"], params["min_coverage_pct"].strip()))
        cov = None
    settings["min_coverage_pct"] = cov

    # --- explicit instrument drops ----------------------------------------
    # One argv token can legitimately contain spaces ("RX-102, RX-103"), so
    # split on commas and strip, never on whitespace.
    drops = [d.strip() for d in (params["drop_instruments"] or "").split(",")]
    settings["drop_instruments"] = [d for d in drops if d]

    return settings


def describe_settings(settings):
    # type: (Dict[str, Any]) -> List[str]
    """Human-readable summary of the rules that are actually active."""
    lines = []
    low, high = settings["min_reading"], settings["max_reading"]
    if low is None and high is None:
        # Say the second half out loud: with no bounds the rule still claims
        # rows whose reading is not a number, and a bare "off" would make
        # those look inexplicable in qc_flags.csv.
        lines.append("range rule: off (no bounds) — a reading that is not a "
                     "number is still flagged out_of_range")
    else:
        lines.append("range rule: keep readings in [{}, {}]".format(
            "-inf" if low is None else low, "+inf" if high is None else high))
    lines.append(
        "spike rule: off" if settings["spike_mad"] is None
        else "spike rule: robust |z| > {}".format(settings["spike_mad"]))
    lines.append(
        "flatline rule: off" if settings["flatline_run"] is None
        else "flatline rule: runs of >= {} identical readings".format(
            settings["flatline_run"]))
    lines.append(
        "coverage rule: off" if settings["min_coverage_pct"] is None
        else "coverage rule: drop instruments below {}% coverage".format(
            settings["min_coverage_pct"]))
    lines.append(
        "forced drops: none" if not settings["drop_instruments"]
        else "forced drops: {}".format(", ".join(settings["drop_instruments"])))
    return lines


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------
def find_csvs(data_dir):
    # type: (Path) -> List[Path]
    """Recursively find CSV files under the input tree, extension case-INSENSITIVELY.

    Skips dotfiles/dot-directories and Office lock files (``~$...``), the same
    hygiene rule the other capsules in this demo apply.

    The extension is matched case-insensitively because ``rglob("*.csv")`` is
    not: pathlib's pattern matching is case-SENSITIVE on POSIX whatever the
    filesystem underneath does, and Code Ocean runs these capsules on a
    case-sensitive one. A file exported as ``INSTRUMENTS.CSV`` was therefore
    invisible to this capsule — absent from ``input_files``, never a candidate
    for the readings file, never passed through into ``/results``, and with
    nothing in the log or the manifest to say a file had been passed over. That
    is data silently lost from the captured result asset, and it could go
    either way depending on the machine: the same asset is seen on a
    developer's case-insensitive laptop and skipped in the cloud. Matching on
    ``suffix.lower()`` makes the answer the same everywhere.
    """
    if not data_dir.is_dir():
        return []
    found = []
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() != ".csv":
            continue
        if any(part.startswith(".") for part in p.relative_to(data_dir).parts):
            continue
        if p.name.startswith("~$"):
            continue
        if p.is_file():
            found.append(p)
    return found


def normalize_header(name):
    # type: (Any) -> str
    """The canonical form of one column name: trimmed and lower-cased."""
    return str(name).strip().lower()


def drop_duplicate_columns(df):
    # type: (pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]
    """Keep the FIRST column of each normalized name, drop the later ones.

    Every column lookup in this capsule is by TRIMMED, LOWER-CASED name, which
    is what lets it accept ``Timestamp`` as ``timestamp`` — and which means two
    distinct columns can collapse onto one name. ``Reading,reading``,
    ``Timestamp,timestamp``, ``reading ,reading`` and
    ``Instrument_ID,instrument_id`` are all pairs pandas keeps apart (they are
    different strings) that this capsule then treats as one.

    Left alone, that is silent data selection: the old lookup table was built
    with a dict comprehension, so the LAST colliding column quietly won, no
    warning was raised, and a duplicate ``reading`` column full of rubbish
    could take over the whole QC pass with an empty ``warnings`` list in the
    manifest. It is also a live crash for the next step, which does
    ``df[["timestamp", "instrument_id", "reading"]]`` and gets a two-column
    frame back where it expected a Series.

    So the collision is resolved here, at read time, where it can still be
    explained: the first column of each name wins (it is the one a reader
    scanning the header left-to-right would expect), the rest are dropped
    before any rule reads them and before anything is written, and the choice
    is returned so the caller can log it, warn about it and record it in the
    manifest. It is never silent, and it never ends the run.

    Returns (frame with unique normalized names, one record per dropped column).
    """
    seen = {}  # type: Dict[str, str]
    keep_positions = []  # type: List[int]
    dropped = []  # type: List[Dict[str, Any]]
    for position, name in enumerate(df.columns):
        key = normalize_header(name)
        if key in seen:
            dropped.append({
                "column": str(name),
                "position": position,
                "normalized_name": key,
                "kept_column": seen[key],
            })
        else:
            seen[key] = str(name)
            keep_positions.append(position)
    if not dropped:
        return df, []
    # .iloc, not [], because the labels are exactly what is ambiguous here.
    return df.iloc[:, keep_positions].copy(), dropped


def duplicate_columns_warning(source_name, dropped):
    # type: (str, List[Dict[str, Any]]) -> str
    """The warning text for columns that collided once headers were normalized."""
    pairs = "; ".join(
        "'{}' (column {}) duplicates '{}' as '{}'".format(
            d["column"], d["position"] + 1, d["kept_column"], d["normalized_name"])
        for d in dropped)
    return (
        "{} has {} column name(s) that collide once trimmed and lower-cased "
        "({}) — the FIRST column of each name was kept and the duplicate(s) "
        "were dropped before any QC rule read them, so they are absent from "
        "clean_readings.csv and qc_flags.csv too".format(
            source_name, len(dropped), pairs))


def read_csv_safely(path):
    # type: (Path) -> Tuple[Optional[pd.DataFrame], List[Dict[str, Any]], Optional[str]]
    """Read one CSV. Unreadable -> ``(None, [], "why")``, never an exception.

    Column names that collide once normalized are resolved here rather than
    left to detonate later; the second return value says what was dropped so
    the caller can report it (see ``drop_duplicate_columns``).

    The THIRD return value is why the file could not be read, and it exists
    because returning a bare ``None`` made a whole input file disappear in
    silence. A zero-byte CSV, a binary file with a ``.csv`` name, or one the
    mount denies us permission to open was logged once and then dropped on the
    floor: it stayed in the manifest's ``input_files`` (so it looked consumed),
    it never reached ``others`` (so it was never copied into ``/results``), and
    ``warnings``/``notes`` were both empty. For a lookup table like
    ``instruments.csv`` that means the file silently fails to travel with the
    captured result asset while the manifest positively implies it did. The
    error text is handed back so the caller can put the file in ``warnings``
    AND in the manifest, by name and with the reason.
    """
    try:
        df = pd.read_csv(str(path))
    except Exception as exc:  # noqa: BLE001 - an unreadable sidecar is not fatal
        log("warning: could not read {} ({})".format(path.name, exc))
        return None, [], "{}: {}".format(type(exc).__name__, exc)
    kept, dropped = drop_duplicate_columns(df)
    return kept, dropped, None


def qc_columns_in(df):
    # type: (pd.DataFrame) -> List[str]
    """Which QC verdict columns (if any) this frame already carries."""
    cols = set(str(c).strip().lower() for c in df.columns)
    return [c for c in QC_OUTPUT_COLUMNS if c in cols]


def drop_qc_columns(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    """Remove stale qc_status/qc_reason so this run writes the only verdicts."""
    stale = [c for c in df.columns if str(c).strip().lower() in QC_OUTPUT_COLUMNS]
    return df.drop(columns=stale) if stale else df


def rel_to(path, data_dir):
    # type: (Path, Path) -> str
    """One input file's path as the manifest names it (relative to the mount)."""
    try:
        return str(path.relative_to(data_dir))
    except ValueError:  # not under data_dir at all — name it as best we can
        return path.name


def choose_by_name_then_size(candidates):
    # type: (List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]
    """A canonical filename wins; largest-file-wins is only the tie-breaker.

    Returns (chosen entry, why it was chosen). The reason is handed back rather
    than only logged because it is quoted in the manifest against every
    candidate that LOST — "which file did these numbers come from, and what
    else was in the mount" has to be answerable from the artifact alone. Step 2
    resolves the same mount with the same rule and records the same thing.
    """
    for name in CANONICAL_READINGS_NAMES:
        for candidate in candidates:
            if candidate["path"].name.lower() == name:
                return candidate, "canonical name {}".format(name)
    ordered = sorted(candidates,
                     key=lambda c: (c["path"].stat().st_size, str(c["path"])))
    return ordered[-1], "no canonically named readings file ({}), so the " \
        "largest file won".format("/".join(CANONICAL_READINGS_NAMES))


def pick_readings_file(csvs, data_dir, notes):
    # type: (List[Path], Path, List[str]) -> Dict[str, Any]
    """Choose the readings file; everything else is passed through untouched.

    Selection order, and the order matters:

    1. a CSV must have a ``reading`` column to be a candidate at all;
    2. candidates that already carry ``qc_status``/``qc_reason`` are SKIPPED
       with a log line — those are a previous QC run's output, and re-filtering
       one would bake its stale verdicts into this run's "clean" readings;
    3. a canonical name wins: ``readings.csv``, then ``clean_readings.csv``;
    4. only then, largest-file-wins.

    Size alone is the wrong rule, which is the bug this ordering fixes:
    ``qc_flags.csv`` is every input row PLUS two verdict columns, so it is
    bigger than the ``clean_readings.csv`` beside it. If a step-1 result asset
    is ever re-mounted into step 1, largest-wins would silently pick the flags
    file and emit a "clean" CSV carrying old QC columns, with exit 0.

    Last resort: if EVERY candidate is a QC output, one of them is used anyway
    (an empty pipeline is a worse answer than a re-filtered one) — but its
    stale QC columns are dropped first, so this run's verdicts are the only
    ones in the output. That is recorded in the manifest's ``notes``.

    A CSV that cannot be READ at all is the fifth case, and it used to have no
    case: ``read_csv_safely`` logged one line and this loop went straight on to
    the next file, so the file was neither a candidate nor an "other". It
    therefore never reached ``passthrough``, never appeared in ``outputs``, and
    left ``warnings``/``notes`` empty — while ``input_files`` still listed it,
    which reads as "consumed". Unreadable files are now collected and returned
    so the caller can name every one of them in ``warnings`` and in the
    manifest; they are ALSO appended to ``others`` so ``passthrough`` still
    tries to copy the bytes across (QC has no opinion about a file it could not
    parse, and a lookup table that merely confuses pandas should still travel
    with the dataset). ``passthrough`` reports whichever copies fail.

    A sixth case is not an error at all and was silent for exactly that
    reason: a readings-shaped file that simply LOST the selection above. It was
    passed through to /results, so no bytes were lost — but nothing in the log
    or the manifest said that the QC verdicts describe one file out of several,
    and step 2 (which used to pick by a different rule entirely) could well
    have analysed a different one. Every unchosen candidate is therefore
    recorded by name, row count and reason, under the same manifest key step 2
    uses for the same mount.

    Returns a dict: ``path``, ``frame``, ``others`` (everything to pass
    through), ``dropped_columns`` (of the CHOSEN file only — the pass-through
    files are copied byte for byte, so nothing was dropped from them),
    ``unreadable`` as (path, error) pairs, ``not_chosen`` records, and
    ``chosen_reason``.
    """
    candidates = []  # type: List[Dict[str, Any]]
    qc_outputs = []  # type: List[Dict[str, Any]]
    others = []  # type: List[Path]
    unreadable = []  # type: List[Tuple[Path, str]]

    for path in csvs:
        df, dropped, error = read_csv_safely(path)
        if df is None:
            # Recorded, not swallowed — and still offered to passthrough, so
            # the bytes survive even though this step could not parse them.
            unreadable.append((path, error or "unreadable"))
            others.append(path)
            continue
        cols = set(normalize_header(c) for c in df.columns)
        if READING_COL not in cols:
            others.append(path)  # e.g. instruments.csv — a lookup table
            continue
        stale = qc_columns_in(df)
        entry = {"path": path, "file": rel_to(path, data_dir), "frame": df,
                 "dropped": dropped, "rows": int(len(df)), "qc_columns": stale}
        if stale:
            log("skipping {} as the readings file: it already carries {} — "
                "that is a previous QC run's output, not raw readings".format(
                    path.name, "/".join(stale)))
            qc_outputs.append(entry)
        else:
            candidates.append(entry)

    if not candidates and not qc_outputs:
        return {"path": None, "frame": None, "others": sorted(others),
                "dropped_columns": [], "unreadable": unreadable,
                "not_chosen": [], "chosen_reason": None}

    last_resort = not candidates
    pool = candidates if candidates else qc_outputs
    if last_resort:
        log("every CSV with a '{}' column is a QC output — re-filtering one "
            "rather than emitting nothing".format(READING_COL))

    chosen, reason = choose_by_name_then_size(pool)
    log("readings file chosen by {}: {}".format(reason, chosen["file"]))

    frame = chosen["frame"]
    if last_resort:
        notes.append(
            "every CSV with a '{}' column already carried QC verdict columns; "
            "used {} anyway and dropped its stale {} column(s) before "
            "filtering".format(
                READING_COL, chosen["path"].name, "/".join(QC_OUTPUT_COLUMNS)))
        log("warning: {}".format(notes[-1]))
        frame = drop_qc_columns(frame)

    not_chosen = []  # type: List[Dict[str, Any]]
    for entry in candidates + qc_outputs:
        if entry["path"] == chosen["path"]:
            continue
        others.append(entry["path"])
        is_qc_output = bool(entry["qc_columns"]) and not last_resort
        if is_qc_output:
            why = ("it carries {} — a previous QC run's output rather than raw "
                   "readings".format("/".join(entry["qc_columns"])))
        else:
            why = "{} was preferred ({})".format(chosen["file"], reason)
        not_chosen.append({"file": entry["file"], "rows": entry["rows"],
                           "reason": why,
                           # true = a re-mounted QC output, which this step
                           # always declines by design, rather than a rival
                           # readings file. The caller uses it to decide
                           # warning vs note; the record is the same either way.
                           "is_qc_output": is_qc_output})

    return {"path": chosen["path"], "frame": frame, "others": sorted(others),
            "dropped_columns": chosen["dropped"], "unreadable": unreadable,
            "not_chosen": not_chosen, "chosen_reason": reason}


def not_chosen_readings_warning(readings_file, not_chosen):
    # type: (Optional[str], List[Dict[str, Any]]) -> str
    """The warnings entry naming every readings-shaped file that LOST the pick.

    Nothing is wrong with these files and nothing was lost — they are copied
    into /results like any other pass-through input. What was missing is the
    sentence that makes the QC verdicts interpretable: they describe ONE of the
    readings files that were mounted. Step 2 records the same list, under the
    same key, so the two manifests can be laid side by side and seen to be
    talking about the same file.
    """
    listed = "; ".join(
        "{} ({} row(s)) — {}".format(r["file"], r["rows"], r["reason"])
        for r in not_chosen)
    return (
        "{} other file(s) in the input have a '{}' column and were NOT "
        "QC-filtered: {}. Every verdict in clean_readings.csv, qc_flags.csv "
        "and qc_summary.csv comes from {} alone; the others were handed to the "
        "pass-through step untouched (see passthrough_files, and "
        "input_files_not_copied for any the copy could not take)".format(
            len(not_chosen), READING_COL, listed, readings_file))


def unreadable_files_warning(entries):
    # type: (List[Dict[str, Any]]) -> str
    """The warnings entry naming every input CSV that could not be read."""
    listed = "; ".join(
        "{} ({}{})".format(
            e["file"], e["error"],
            "" if e["passed_through"]
            else "; NOT copied into the results either")
        for e in entries)
    return (
        "{} input CSV file(s) could not be read and contributed NOTHING to "
        "this run's QC verdicts — they are listed in input_files because they "
        "were found, and in unreadable_input_files with the reason: {}".format(
            len(entries), listed))


def note_unreadable_inputs(records, copied, warnings):
    # type: (List[Dict[str, Any]], List[str], List[str]) -> None
    """Finish the unreadable-input records and put them in ``warnings``.

    Called after ``passthrough``, because "could pandas parse it" and "did the
    bytes reach /results" are two different answers and the manifest owes the
    reader both. Without this, an unreadable file left no trace anywhere.
    """
    if not records:
        return
    copied_set = set(copied)
    for record in records:
        record["passed_through"] = record["file"] in copied_set
    warnings.append(unreadable_files_warning(records))
    log("warning: {}".format(warnings[-1]))


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------
# Timestamps arrive as text and are never trusted. Two things go wrong with
# them often enough to have ended a run with exit 1 and a COMPLETELY EMPTY
# /results, and both of them are DATA problems, so neither may stop the run:
#
#   MIXED UTC OFFSETS. A real export can carry ``+00:00`` on one row and
#   ``+05:30`` on the next. Plain ``pd.to_datetime(errors="coerce")`` cannot
#   choose one dtype for that and hands back an OBJECT-dtype Index of per-row
#   datetimes — a FutureWarning, not an error. ``errors="coerce"`` is no
#   protection here: it coerces per-ELEMENT parse failures, it says nothing
#   about the dtype of the RESULT. This capsule limped along on the object
#   column, exited 0 with an EMPTY `warnings` list, and copied the offsets
#   verbatim into clean_readings.csv — where step 2 could not resample them and
#   died with "Only valid with DatetimeIndex ... got an instance of 'Index'".
#   Parsing with ``utc=True`` and then dropping the zone puts every row on one
#   timeline in a single tz-naive ``datetime64`` dtype, so coverage and the
#   flatline ordering compare real instants rather than wall-clock text. An
#   all-naive column round-trips unchanged. Step 2's ``analyze.py`` does the
#   identical thing, and so does batch 1's ``make_report.py``.
#
#   IMPLAUSIBLE INSTANTS. ``1700-01-01T00:00:00`` is inside pandas'
#   representable range, so it parses cleanly — and then ``max - min`` against
#   ordinary 2026 data is a span of ~326 years, which overflows an int64 count
#   of NANOSECONDS at ~292 years and raises OutOfBoundsDatetime. One sentinel
#   row among thirteen was enough to write nothing at all. Such a value is
#   still data, so the row is still QC'd on its reading, still written to
#   clean_readings.csv and qc_flags.csv, and still counted in qc_summary.csv;
#   it is only kept out of the SPAN arithmetic behind coverage_pct, exactly as
#   an unparseable timestamp already is, and the count is reported as
#   ``n_implausible_timestamps``.
#
# ``span_seconds`` is ALSO made arithmetically overflow-proof, because a filter
# that only catches the cases we thought of is not the same as arithmetic that
# cannot fail.
PLAUSIBLE_MIN_TIMESTAMP = pd.Timestamp("1900-01-01")
PLAUSIBLE_MAX_TIMESTAMP = pd.Timestamp("2200-01-01")

# A trailing UTC offset ("+05:30", "-0400") or the "Z" zone marker — how a raw
# value announces that it carries a zone at all. Matched against the ORIGINAL
# text, because after parsing to UTC there is nothing left to count.
UTC_OFFSET_PATTERN = r"(?:[Zz]|[+-]\d{2}:?\d{2})\s*$"

# How a timestamp value is SHAPED, which decides how it gets interpreted. Only
# the first of these is unambiguous; see describe_timestamp_format for what the
# other two silently commit the run to. Step 2 uses the same three patterns and
# reports the same block, so one input is described identically by both.
ISO_TIMESTAMP_PATTERN = (
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"\s*(?:[Zz]|[+-]\d{2}:?\d{2})?$")
NUMERIC_TIMESTAMP_PATTERN = r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$"
DAY_MONTH_TIMESTAMP_PATTERN = r"^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"


def _mixed_timestamp_format_supported():
    # type: () -> bool
    """Can this pandas parse a column of per-row date formats (``format="mixed"``)?

    Feature-probed rather than version-sniffed, and probed with the exact shape
    that matters: one offset-AWARE value next to one offset-NAIVE value.

    ``format="mixed"`` arrived in pandas 2.0. On pandas 1.x the keyword exists
    but "mixed" is taken as a literal strftime pattern, which with
    ``errors="coerce"`` would quietly return an all-NaT column — so the probe
    uses ``errors="raise"``, where 1.x fails loudly and we fall back.
    """
    try:
        pd.to_datetime(
            pd.Series(["2025-01-02T00:00:00+00:00", "2025-01-03"]),
            format="mixed", utc=True, errors="raise",
        )
    except Exception:  # noqa: BLE001 - any failure means "not supported here"
        return False
    return True


#: Probed once at import; used by _to_utc_datetime().
MIXED_TIMESTAMP_FORMAT_SUPPORTED = _mixed_timestamp_format_supported()


def _to_utc_datetime(values):
    # type: (pd.Series) -> pd.Series
    """``pd.to_datetime(values, utc=True)``, per-row format where pandas allows it.

    ``format="mixed"`` makes pandas infer a format for EVERY element instead of
    inferring one from the first and coercing everything that disagrees. On a
    pandas too old to support it (probed above) this degrades to the plain
    single-format parse — ``utc=True`` is the part that guarantees one dtype,
    and it is present either way.
    """
    if MIXED_TIMESTAMP_FORMAT_SUPPORTED:
        try:
            return pd.to_datetime(
                values, errors="coerce", utc=True, format="mixed")
        except Exception:  # noqa: BLE001 - fall through to the plain parse
            pass
    return pd.to_datetime(values, errors="coerce", utc=True)


def utc_offsets_in(values):
    # type: (pd.Series) -> Tuple[int, List[str]]
    """How many RAW values carry an explicit UTC offset, and which offsets.

    Counted before parsing, because ``utc=True`` is precisely what erases the
    evidence. More than one distinct entry here is the mixed-offset case that
    has no single dtype.
    """
    tz = getattr(values.dtype, "tz", None)
    if tz is not None:  # already tz-aware (not reachable from read_csv)
        return int(values.notna().sum()), [str(tz)]
    if values.dtype != object:
        return 0, []
    text = values.astype(str).str.strip()
    found = text.str.extract("(" + UTC_OFFSET_PATTERN + ")", expand=False)
    found = found.dropna().str.strip().str.upper()
    return int(len(found)), sorted(set(found.tolist()))


def empty_timestamp_format():
    # type: () -> Dict[str, Any]
    """The "no timestamp column was examined" shape of the format report.

    ``unambiguous_iso8601`` is null rather than true on purpose: with nothing
    to look at — an input with no ``timestamp`` column at all is a case this
    capsule explicitly tolerates — "the column was unambiguous" would be an
    assertion this run is in no position to make.
    """
    return {
        "unambiguous_iso8601": None,
        "n_iso8601": 0,
        "n_numeric": 0,
        "n_day_month_ambiguous": 0,
        "n_other_text": 0,
        "n_blank": 0,
        "examples": {},
        "examples_read_as": {},
        "per_element_format_inference": MIXED_TIMESTAMP_FORMAT_SUPPORTED,
    }


def describe_timestamp_format(values):
    # type: (pd.Series) -> Dict[str, Any]
    """Classify how the RAW timestamp text will be INTERPRETED, before parsing.

    Reading a timestamp column is a reading of the data, and when the text is
    not unambiguous ISO-8601 that reading is a guess this capsule makes on the
    operator's behalf — silently, until now. Two guesses in particular change
    this step's verdicts:

      * a BARE NUMBER. ``1780000000`` is a perfectly ordinary epoch-SECONDS
        value, and ``pd.to_datetime(..., utc=True)`` reads a numeric column as
        NANOSECONDS: the whole series lands in the first two seconds of 1970.
        ``coverage_pct`` is then measured over a span of milliseconds and the
        flatline rule walks a different ordering, with nothing anywhere saying
        the column was read that way.
      * a DAY/MONTH/YEAR-style date. ``format="mixed"`` infers the format per
        ELEMENT, so ``01/02/2026`` (ambiguous, read month-first) and
        ``13/02/2026`` (unambiguous, read day-first) in the SAME column are
        read with DIFFERENT conventions. That is not a parse failure — it
        succeeds and hands back dates in two different calendars.

    Neither is something this capsule can fix (the file does not say which
    convention it meant) and neither is a reason to fail: they are data. What
    they are is a fact about the result, so the counts and a few example values
    go into the manifest under ``timestamp_interpretation`` and into a warning
    in words. Step 2 reports the identical block for the identical input.

    Blank/NaT-ish values are counted separately and do NOT make the column
    ambiguous: they are already accounted for as unusable timestamps.
    """
    fmt = empty_timestamp_format()
    if len(values) == 0:
        fmt["unambiguous_iso8601"] = True
        return fmt
    text = values.astype(str).str.strip()
    blank = (values.isna() | text.eq("")
             | text.str.lower().isin(["nan", "nat", "none", "null", "na"]))
    iso = ~blank & text.str.match(ISO_TIMESTAMP_PATTERN)
    numeric = ~blank & ~iso & text.str.match(NUMERIC_TIMESTAMP_PATTERN)
    day_month = (~blank & ~iso & ~numeric
                 & text.str.match(DAY_MONTH_TIMESTAMP_PATTERN))
    other = ~blank & ~iso & ~numeric & ~day_month

    fmt["n_iso8601"] = int(iso.sum())
    fmt["n_numeric"] = int(numeric.sum())
    fmt["n_day_month_ambiguous"] = int(day_month.sum())
    fmt["n_other_text"] = int(other.sum())
    fmt["n_blank"] = int(blank.sum())
    fmt["unambiguous_iso8601"] = bool(
        fmt["n_numeric"] == 0 and fmt["n_day_month_ambiguous"] == 0
        and fmt["n_other_text"] == 0)
    for key, mask in (("numeric", numeric),
                      ("day_month_ambiguous", day_month),
                      ("other_text", other)):
        found = [str(v) for v in text[mask].drop_duplicates().head(3).tolist()]
        if found:
            fmt["examples"][key] = found
    return fmt


def annotate_parsed_examples(fmt, values, parsed):
    # type: (Dict[str, Any], pd.Series, pd.Series) -> Dict[str, Any]
    """Record what each example value was ACTUALLY read as.

    Taken from this run's own parsed column rather than re-parsed here, because
    the answer depends on the column's dtype as well as on the text: a column
    of integers is read as nanoseconds since the epoch, while the same digits
    as strings among other text are read as NaT. An illustration that guessed
    would be one more claim the manifest cannot support.
    """
    text = values.astype(str).str.strip()
    read_as = {}  # type: Dict[str, List[str]]
    for kind, samples in fmt["examples"].items():
        rendered = []
        for sample in samples:
            hits = parsed[text == sample].dropna()
            rendered.append("\"{}\" -> {}".format(
                sample,
                pd.Timestamp(hits.iloc[0]).isoformat() if len(hits)
                else "NaT (unusable)"))
        read_as[kind] = rendered
    fmt["examples_read_as"] = read_as
    return fmt


def timestamp_format_warning(fmt, column, consequence):
    # type: (Dict[str, Any], str, str) -> str
    """The warning for a timestamp column that is not unambiguous ISO-8601."""
    read_as = fmt.get("examples_read_as", {})
    parts = []  # type: List[str]
    if fmt["n_numeric"]:
        parts.append(
            "{} value(s) are bare numbers, which a numeric column reads as "
            "NANOSECONDS since 1970-01-01, so an epoch value in seconds or "
            "milliseconds lands in 1970 rather than where it was meant to "
            "({})".format(fmt["n_numeric"],
                          "; ".join(read_as.get("numeric", []))))
    if fmt["n_day_month_ambiguous"]:
        parts.append(
            "{} value(s) are day/month/year-style dates whose day-and-month "
            "order is inferred PER VALUE, so \"01/02/2026\" and \"13/02/2026\" "
            "in one column are read with DIFFERENT conventions — month-first "
            "for the first, day-first for the second ({})".format(
                fmt["n_day_month_ambiguous"],
                "; ".join(read_as.get("day_month_ambiguous", []))))
    if fmt["n_other_text"]:
        parts.append(
            "{} value(s) are some other text, left to pandas' general date "
            "parser and read as NaT if it cannot make sense of them "
            "({})".format(fmt["n_other_text"],
                          "; ".join(read_as.get("other_text", []))))
    return (
        "the '{}' column is NOT unambiguous ISO-8601, so how it was read is an "
        "interpretation and not a fact about the file: {}. {} The counts and "
        "examples are in the manifest under timestamp_interpretation; supply "
        "ISO-8601 timestamps (2026-06-01T00:00:00) if the reading above is not "
        "the one you meant. clean_readings.csv and qc_flags.csv still carry "
        "the original text unchanged".format(
            column, "; ".join(parts), consequence))


def empty_timestamp_info():
    # type: () -> Dict[str, Any]
    """The "nothing to report" shape of ``parse_timestamp_series``'s info dict.

    So the manifest carries the same keys whether or not there was a timestamp
    column at all — a reader must never have to tell "none of this happened"
    from "this run did not report it".
    """
    return {
        "n_with_offset": 0,
        "offsets": [],
        "n_implausible": 0,
        "implausible": None,
        "dtype_fallback": None,
        "format": empty_timestamp_format(),
    }


def parse_timestamp_series(values):
    # type: (pd.Series) -> Tuple[pd.Series, Dict[str, Any]]
    """Parse a timestamp column to ONE tz-naive ``datetime64`` dtype.

    Returns ``(parsed, info)``. ``info`` reports everything the parse changed
    or noticed, so the caller can put it in the run log AND the manifest:

      n_with_offset   raw values that carried an explicit UTC offset or "Z"
      offsets         the distinct offsets seen (>1 entry = the mixed case)
      n_implausible   parsed values outside [1900-01-01, 2200-01-01)
      implausible     the boolean mask for those rows, aligned to ``values``
      dtype_fallback  set only if the parse somehow did not yield a datetime
                      dtype, in which case the column degrades to all-NaT
      format          how the raw text was INTERPRETED, and whether that was
                      an interpretation at all (see describe_timestamp_format)

    The dtype check is not decoration. ``errors="coerce"`` protects against a
    single unreadable ELEMENT; it does not promise a datetime-typed RESULT, and
    an object-dtype column reaches ``.dt``, ``.resample`` and the span
    arithmetic looking fine and then raises. Checking here means the rest of
    the capsule can treat the column as ``datetime64`` without another guard —
    and if a future pandas ever breaks that promise, the run degrades to "no
    usable timestamps" with a reported reason instead of writing nothing.
    """
    info = empty_timestamp_info()
    info["n_with_offset"], info["offsets"] = utc_offsets_in(values)
    # How the text will be READ, worked out before parsing erases the evidence.
    # A column that is not unambiguous ISO-8601 is interpreted, not merely
    # read, and the interpretation belongs in the manifest.
    info["format"] = describe_timestamp_format(values)

    parsed = _to_utc_datetime(values)
    if getattr(parsed.dtype, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        info["dtype_fallback"] = str(parsed.dtype)
        parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    # What each example value was actually read AS — from this run's own parse,
    # so the manifest illustrates the interpretation instead of guessing it.
    info["format"] = annotate_parsed_examples(info["format"], values, parsed)

    implausible = parsed.notna() & (
        (parsed < PLAUSIBLE_MIN_TIMESTAMP) | (parsed >= PLAUSIBLE_MAX_TIMESTAMP))
    info["implausible"] = implausible
    info["n_implausible"] = int(implausible.sum())
    return parsed, info


def span_seconds(timestamps):
    # type: (pd.Series) -> float
    """``max - min`` in seconds, computed so that it CANNOT overflow.

    The obvious ``(valid.max() - valid.min()).total_seconds()`` raises
    ``OutOfBoundsDatetime``/``OverflowError`` as soon as the span passes ~292
    years, even though BOTH endpoints are perfectly representable — a single
    1700-01-01 sentinel among 2026 data was enough to exit 1 with nothing
    written.

    The fix is to do the subtraction in MICROSECONDS. An int64 count of
    microseconds spans +-292,471 years, so no pair of instants pandas can
    represent at all (1677..2262) can overflow it, while nanoseconds overflow
    well inside pandas' own representable range. Truncating below a microsecond
    cannot move an hourly coverage figure. Fewer than two usable timestamps
    means there is no span, which is 0 rather than an error.
    """
    valid = timestamps.dropna()
    if len(valid) < 2:
        return 0.0
    micros = valid.to_numpy(dtype="datetime64[us]").astype("int64")
    # int() first: the subtraction happens in Python's unbounded integers, so
    # there is no width to overflow even before the microsecond argument.
    return float(int(micros.max()) - int(micros.min())) / 1e6


# ---------------------------------------------------------------------------
# QC rules
# ---------------------------------------------------------------------------
def coverage_percent(timestamps):
    # type: (pd.Series) -> Tuple[float, Optional[str]]
    """Observed readings / expected hourly slots between first and last, in %.

    A gap of missing hours pushes this down, which is exactly the "instrument
    was offline for a while" signal the coverage rule looks for. With no usable
    timestamps (or a single one) there is nothing to be missing, so 100.

    The span is measured by ``span_seconds``, which cannot overflow. Doing it
    the obvious way — ``(valid.max() - valid.min()).total_seconds()`` — raised
    ``OutOfBoundsDatetime`` on any span over ~292 years and ended the run with
    an empty ``/results``; the caller additionally keeps implausible instants
    out of ``timestamps`` altogether (see ``classify``), but this function does
    not rely on that.

    Returns (percent, explanation-or-None). The explanation exists because the
    ratio is only a coverage figure while the readings really are hourly: a
    denser-than-hourly instrument, and above all one whose timestamps all
    REPEAT (span 0 hours -> one expected slot), divides by a number smaller
    than the row count and reports something like ``1200.0`` %. A coverage
    percentage above 100 is not a measurement, it is an artefact of the hourly
    assumption, and printing it into qc_summary.csv with nothing to explain it
    is exactly the "manifest claims more than it knows" defect. So it is
    clamped to 100 — the honest answer, since nothing is missing — and the
    reason is handed back for the caller to record.
    """
    valid = timestamps.dropna()
    if len(valid) < 2:
        return 100.0, None
    span_hours = span_seconds(valid) / 3600.0
    expected = int(round(span_hours)) + 1
    if expected <= 0:
        return 100.0, None
    raw = round(100.0 * len(valid) / expected, 2)
    if raw <= 100.0:
        return raw, None
    return 100.0, (
        "{} reading(s) fall in {} expected hourly slot(s), which works out at "
        "{}% — coverage is measured against an HOURLY grid, so readings that "
        "repeat a timestamp or arrive faster than hourly can exceed 100%. "
        "{}Nothing is missing, so coverage_pct is reported as 100.0".format(
            len(valid), expected, raw,
            "Every usable timestamp here is identical (span 0 hours). "
            if span_hours == 0 else ""))


def flatline_mask(values, run_length):
    # type: (np.ndarray, int) -> np.ndarray
    """Mark every element that belongs to a run of >= run_length identical values.

    ``values`` must already be in time order. NaN never equals NaN, so missing
    readings break a run rather than extending it.
    """
    mask = np.zeros(len(values), dtype=bool)
    start = 0
    for i in range(1, len(values) + 1):
        same = i < len(values) and values[i] == values[i - 1]
        if not same:
            if i - start >= run_length:
                mask[start:i] = True  # flag the WHOLE run, not just the repeats
            start = i
    return mask


def robust_z(values, reference):
    # type: (np.ndarray, np.ndarray) -> Optional[np.ndarray]
    """Robust z-score of ``values`` against the median/MAD of ``reference``.

    ``reference`` deliberately excludes the out-of-range rows: leaving a 300 °C
    reading in would inflate MAD enough for the moderate spikes to look normal.
    Returns None when MAD is 0 (a perfectly flat reference has no scale, so the
    spike rule cannot say anything).
    """
    ref = reference[~np.isnan(reference)]
    if len(ref) == 0:
        return None
    median = float(np.median(ref))
    mad = float(np.median(np.abs(ref - median)))
    if mad <= 0:
        return None
    return MAD_SCALE * (values - median) / mad


def classify(df, settings, warnings, notes):
    # type: (pd.DataFrame, Dict[str, Any], List[str], List[str]) -> Tuple[np.ndarray, pd.Series, Dict[str, float], Dict[str, int]]
    """Assign exactly one qc_reason to every input row.

    Returns (reasons array aligned to df's positions, instrument id series,
    coverage % per instrument, per-run counts for the manifest).

    Some of those counts exist because the manifest used to contradict the
    data about them, and each is a SUBSET of numbers already reported — extra
    detail, never an extra bucket, so the row accounting in qc_summary.csv
    still balances:

      n_unparseable_readings    rows whose `reading` is not a number. They are
                                counted under `out_of_range` (see below).
      n_unusable_timestamps     rows whose `timestamp` could not be parsed.
                                They are still QC'd on their reading, but they
                                cannot contribute to coverage_pct.
      n_implausible_timestamps  rows whose `timestamp` PARSES but lands outside
                                [PLAUSIBLE_MIN_TIMESTAMP,
                                PLAUSIBLE_MAX_TIMESTAMP). Treated exactly like
                                the line above — still QC'd, still written with
                                their original text, still counted, but kept
                                out of coverage_pct, because one of them turns
                                a 12-hour span into a 326-year one and used to
                                overflow int64 nanoseconds outright.
    """
    n = len(df)
    reasons = np.array(["ok"] * n, dtype=object)

    # Work on normalized copies; the output keeps the original columns as-is.
    lower = dict((str(c).strip().lower(), c) for c in df.columns)
    readings = pd.to_numeric(df[lower[READING_COL]], errors="coerce").to_numpy(dtype=float)

    if TIMESTAMP_COL in lower:
        # One tz-naive datetime64 dtype, whatever zones the text carried; see
        # parse_timestamp_series for why the DTYPE, not just the elements, is
        # what has to be guaranteed here.
        stamps, ts_info = parse_timestamp_series(df[lower[TIMESTAMP_COL]])
        # errors="coerce" turns a typo, a blank, or a date outside pandas'
        # representable range into NaT — and NaT rows silently vanish from the
        # coverage arithmetic. Reporting nothing let a single usable row give
        # "coverage_pct 100.0" with an empty warnings list, which reads as a
        # healthy instrument. Step 2 already reports its own rows_unusable;
        # this is step 1's equivalent.
        n_bad_stamps = int(stamps.isna().sum())
        if n_bad_stamps:
            warnings.append(
                "{} of {} row(s) have a '{}' that could not be parsed — those "
                "rows are excluded from coverage_pct (an instrument left with "
                "fewer than two parseable timestamps reports 100%) and sort "
                "last within their instrument for the flatline rule; they are "
                "still QC'd and kept or dropped on their reading "
                "alone".format(n_bad_stamps, n, TIMESTAMP_COL))
        if ts_info["n_with_offset"]:
            # Reading a zone-qualified value as an INSTANT changes the ordering
            # the flatline rule walks and the span coverage_pct is measured
            # over, so it is a change to this run's verdicts, not a formatting
            # detail. The output columns still carry the operator's original
            # text unaltered — this is how the rules read it, said out loud.
            warnings.append(
                "{} of {} '{}' value(s) carry an explicit UTC offset ({}) — "
                "every timestamp was read as an INSTANT (converted to UTC, "
                "zone dropped) so that coverage_pct and the flatline ordering "
                "compare like with like. A column mixing offsets has no single "
                "datetime dtype at all, which is what used to break the "
                "analysis step. clean_readings.csv and qc_flags.csv still "
                "carry the original text unchanged.".format(
                    ts_info["n_with_offset"], n, TIMESTAMP_COL,
                    ", ".join(ts_info["offsets"])))
        # How the timestamp text was INTERPRETED, when that was a choice rather
        # than a reading. Bare numbers are read as nanoseconds since the epoch
        # and slash-dates have their day/month order inferred per value — both
        # change the ordering the flatline rule walks and the span coverage_pct
        # is measured over, and both used to happen with nothing said anywhere.
        if ts_info["format"]["unambiguous_iso8601"] is False:
            warnings.append(timestamp_format_warning(
                ts_info["format"], TIMESTAMP_COL,
                "coverage_pct and the flatline ordering follow from it, and so "
                "does every timestamp the analysis step reads out of "
                "clean_readings.csv."))
        if ts_info["dtype_fallback"]:
            warnings.append(
                "the '{}' column could not be parsed to a datetime dtype (got "
                "{}) — every row is treated as having an unusable timestamp, "
                "so coverage_pct is 100% everywhere and flatline runs use file "
                "order. No row was dropped for it.".format(
                    TIMESTAMP_COL, ts_info["dtype_fallback"]))
        if ts_info["n_implausible"]:
            warnings.append(
                "{} of {} '{}' value(s) parse but land outside the plausible "
                "range [{} .. {}) — a sentinel or a typo rather than a "
                "measurement. One of them stretches an instrument's span "
                "across centuries, which is not a coverage figure and used to "
                "overflow the int64 nanosecond arithmetic outright. Those rows "
                "are excluded from coverage_pct exactly like an unparseable "
                "timestamp; they are NOT dropped — they are still QC'd on "
                "their reading, still written to clean_readings.csv and "
                "qc_flags.csv, and still counted in qc_summary.csv. See "
                "n_implausible_timestamps.".format(
                    ts_info["n_implausible"], n, TIMESTAMP_COL,
                    PLAUSIBLE_MIN_TIMESTAMP.isoformat(),
                    PLAUSIBLE_MAX_TIMESTAMP.isoformat()))
    else:
        warnings.append(
            "input has no '{}' column — coverage is reported as 100% and "
            "flatline runs use file order".format(TIMESTAMP_COL))
        stamps = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
        ts_info = empty_timestamp_info()
        # No column means no usable timestamp on any row. The warning above
        # already explains it, so this only feeds the manifest's count.
        n_bad_stamps = n

    # The timestamps coverage_pct may measure a span over: the parsed column
    # with the implausible instants blanked out. `stamps` itself is left alone
    # so the flatline rule still sees every row in its real order.
    coverage_stamps = (stamps.mask(ts_info["implausible"])
                       if ts_info["n_implausible"] else stamps)

    n_unknown_instrument = 0
    n_literal_unknown = 0
    if INSTRUMENT_COL in lower:
        instruments = df[lower[INSTRUMENT_COL]].astype(str).str.strip()
        # Rows that carry the substitute label as their REAL id. Counted before
        # anything is relabelled, because afterwards the two groups are one.
        n_literal_unknown = int((instruments == UNKNOWN_INSTRUMENT).sum())
        # Relabelling a row's instrument id is a SUBSTITUTION, and one that
        # changes qc_summary.csv, `instruments_kept` and every downstream
        # grouping. It happened silently: the rows appeared under a plausible
        # "(unknown)" instrument with nothing anywhere saying they had been
        # renamed. Count them here so `warnings` can name them and the manifest
        # can carry `n_unknown_instrument_rows`.
        missing = instruments.isin(MISSING_INSTRUMENT_TOKENS)
        n_unknown_instrument = int(missing.sum())
        if n_unknown_instrument:
            instruments = instruments.mask(missing, UNKNOWN_INSTRUMENT)
            warnings.append(
                "{} of {} row(s) have no '{}' — they were NOT dropped, they "
                "were relabelled to the instrument \"{}\" and QC'd as one "
                "instrument. {}".format(
                    n_unknown_instrument, n, INSTRUMENT_COL,
                    UNKNOWN_INSTRUMENT,
                    "So that id in qc_summary.csv/instruments_kept is this "
                    "capsule's label and not a value from the input file"
                    if not n_literal_unknown else
                    "CAREFUL: {} row(s) of the input already carried the "
                    "literal id \"{}\" themselves, so that label does NOT mean "
                    "\"this capsule's substitute\" here — the two groups "
                    "COLLIDED and were QC'd as one instrument of {} row(s), "
                    "and neither qc_summary.csv nor qc_flags.csv can tell them "
                    "apart. See "
                    "n_rows_with_literal_unknown_instrument_id".format(
                        n_literal_unknown, UNKNOWN_INSTRUMENT,
                        n_literal_unknown + n_unknown_instrument)))
        elif n_literal_unknown:
            # No substitution happened, but `unknown_instrument_label` in the
            # manifest would otherwise invite the opposite misreading: that
            # this capsule invented the instrument. It did not — the file did.
            notes.append(
                "{} row(s) carry the literal '{}' \"{}\", which is also the "
                "label this capsule gives rows with a MISSING '{}'. No row was "
                "relabelled in this run (n_unknown_instrument_rows is 0), so "
                "that instrument is the input's own value".format(
                    n_literal_unknown, INSTRUMENT_COL, UNKNOWN_INSTRUMENT,
                    INSTRUMENT_COL))
            log("note: {}".format(notes[-1]))
    else:
        warnings.append(
            "input has no '{}' column — every row is treated as one "
            "instrument".format(INSTRUMENT_COL))
        instruments = pd.Series(["(all)"] * n, index=df.index)

    # A reading that is not a number is not "outside the bounds" — it is
    # unreadable. It is still counted under out_of_range, because that is the
    # closed six-reason vocabulary this pipeline's contract, its qc_summary
    # columns and its READMEs all share. What must not happen is the manifest
    # reporting the range rule OFF (min_reading/max_reading null, notes saying
    # no bound is applied) while out_of_range rows appear anyway with nothing
    # to explain them. So the count is reported, and the note says plainly
    # that the reason covers unreadable values as well as out-of-bounds ones.
    n_unparseable = int(np.isnan(readings).sum())
    if n_unparseable:
        low, high = settings["min_reading"], settings["max_reading"]
        notes.append(
            "{} reading(s) could not be parsed as a number; they are counted "
            "under the 'out_of_range' reason, which covers unreadable "
            "readings as well as readings outside the bounds{}".format(
                n_unparseable,
                " — that is why out_of_range rows appear even though no "
                "lower or upper bound is in force"
                if low is None and high is None else ""))
        log("note: {}".format(notes[-1]))

    drop_set = set(d.lower() for d in settings["drop_instruments"])
    # Which --drop_instruments values actually hit something, so the ones that
    # hit nothing can be reported instead of looking obeyed (see below).
    drops_matched = {}  # type: Dict[str, List[str]]
    # Instruments whose spike rule could not run. `effective_parameters` says
    # spike_mad is in force for the whole run, so an instrument it silently
    # skipped is the manifest claiming more than it knows.
    spike_skipped = []  # type: List[str]
    coverage = {}  # type: Dict[str, float]
    order = np.arange(n)

    for instrument in sorted(instruments.unique()):
        # Positions of this instrument's rows, in time order (ties keep file
        # order, and rows with an unparseable timestamp sort to the end).
        where = np.flatnonzero((instruments == instrument).to_numpy())
        local = pd.DataFrame({"pos": where, "ts": stamps.to_numpy()[where],
                              "cov_ts": coverage_stamps.to_numpy()[where],
                              "seq": order[where]})
        local = local.sort_values(["ts", "seq"], na_position="last")
        positions = local["pos"].to_numpy()

        cov, cov_note = coverage_percent(pd.Series(local["cov_ts"].to_numpy()))
        coverage[instrument] = cov
        if cov_note:
            # A clamped coverage figure is a substituted value, so it is
            # explained rather than just printed into qc_summary.csv.
            notes.append("{}: {}".format(instrument, cov_note))
            log("note: {}".format(notes[-1]))

        # --- 1. explicit drop ---------------------------------------------
        if instrument.lower() in drop_set:
            drops_matched.setdefault(instrument.lower(), []).append(instrument)
            reasons[positions] = "dropped_instrument"
            log("{}: dropped on request ({} rows)".format(instrument, len(positions)))
            continue

        # --- 2. low coverage ----------------------------------------------
        threshold = settings["min_coverage_pct"]
        if threshold is not None and cov < threshold:
            reasons[positions] = "low_coverage"
            log("{}: coverage {}% < {}% — instrument dropped ({} rows)".format(
                instrument, cov, threshold, len(positions)))
            continue

        values = readings[positions]

        # --- 3. out of range -----------------------------------------------
        # Two different things share this one reason, and the manifest says so
        # (see the n_unparseable note above): a reading OUTSIDE the bounds, and
        # a reading that is not a number at all. The second half fires whether
        # or not a bound is in force — an unreadable value is unusable either
        # way — which is exactly why it has to be reported rather than left to
        # look like the range rule firing with no range.
        low, high = settings["min_reading"], settings["max_reading"]
        unparseable = np.isnan(values)
        outside = np.zeros(len(values), dtype=bool)
        if low is not None:
            outside = outside | (values < low)
        if high is not None:
            outside = outside | (values > high)
        out_of_range = unparseable | outside
        reasons[positions[out_of_range]] = "out_of_range"

        # --- 4. flatline (whole run, in time order) ------------------------
        run_length = settings["flatline_run"]
        if run_length is not None:
            flat = flatline_mask(values, run_length) & ~out_of_range
            reasons[positions[flat]] = "flatline"
        else:
            flat = np.zeros(len(values), dtype=bool)

        # --- 5. spike (robust z against the in-range readings only) --------
        spike_mad = settings["spike_mad"]
        if spike_mad is not None:
            z = robust_z(values, values[~out_of_range])
            if z is None:
                # A rule that is skipped must SAY SO, and the granularity here
                # is the INSTRUMENT: effective_parameters records spike_mad as
                # in force, so an instrument whose MAD is 0 keeps a 60 °C spike
                # as "ok" while the manifest reads as though the rule ran on
                # it. A stdout line alone does not reach the artifact anyone
                # inspects afterwards, so the instrument is collected and
                # reported in warnings and in the manifest.
                spike_skipped.append(instrument)
                log("{}: MAD is 0 (no spread) — spike rule skipped for this "
                    "instrument".format(instrument))
            else:
                spikes = (np.abs(z) > spike_mad) & ~out_of_range & ~flat
                reasons[positions[spikes]] = "spike"

        # --- 6. everything still marked "ok" is kept ------------------------

    # --- what the loop above could not do, said out loud ---------------------
    if spike_skipped:
        warnings.append(
            "the spike rule was SKIPPED for {} of {} instrument(s) ({}): their "
            "in-range readings have a MAD of 0, so a robust z cannot be "
            "computed and no reading of theirs can ever be flagged as a spike. "
            "effective_parameters.spike_mad is still {} — that is the value "
            "the rule WOULD have used, not a rule that ran on every "
            "instrument; see spike_rule_skipped_instruments".format(
                len(spike_skipped), len(coverage), ", ".join(spike_skipped),
                settings["spike_mad"]))

    # A --drop_instruments id that matched nothing is a typo the run should not
    # keep to itself: step 2 already warns when --baseline_instrument names an
    # instrument that is not in the data, and this is the same mistake. The
    # match is case-INSENSITIVE, so a value that matched under a different
    # spelling is worth saying too — the manifest records what the operator
    # typed, and the id that was actually dropped is somewhere else entirely.
    for requested in settings["drop_instruments"]:
        matched = drops_matched.get(requested.lower(), [])
        if not matched:
            warnings.append(
                "{} \"{}\" matched no instrument in the data ({}) — nothing "
                "was dropped on its account".format(
                    PARAM_LABELS["drop_instruments"], requested,
                    ", ".join(sorted(coverage)) or "no instruments at all"))
        else:
            renamed = [m for m in matched if m != requested]
            if renamed:
                notes.append(
                    "{} \"{}\" matched instrument(s) {} (instrument ids are "
                    "compared case-insensitively), and those rows were "
                    "dropped".format(PARAM_LABELS["drop_instruments"],
                                     requested, ", ".join(sorted(renamed))))
                log("note: {}".format(notes[-1]))

    counts = {
        "n_unparseable_readings": n_unparseable,
        "n_unusable_timestamps": n_bad_stamps,
        "n_unknown_instrument_rows": n_unknown_instrument,
        "n_rows_with_literal_unknown_instrument_id": n_literal_unknown,
        "spike_rule_skipped_instruments": sorted(spike_skipped),
        "n_implausible_timestamps": int(ts_info["n_implausible"]),
        "n_timestamps_with_utc_offset": int(ts_info["n_with_offset"]),
        "utc_offsets_seen": list(ts_info["offsets"]),
        "timestamp_interpretation": dict(ts_info["format"]),
    }
    return reasons, instruments, coverage, counts


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def build_summary(instruments, reasons, coverage):
    # type: (pd.Series, np.ndarray, Dict[str, float]) -> pd.DataFrame
    """One row per instrument: counts per reason, coverage and the kept flag."""
    rows = []
    reason_series = pd.Series(reasons, index=instruments.index)
    for instrument in sorted(instruments.unique()):
        mask = (instruments == instrument)
        these = reason_series[mask]
        counts = dict((r, int((these == r).sum())) for r in REASONS)
        n_input = int(mask.sum())
        n_kept = counts["ok"]
        row = {
            INSTRUMENT_COL: instrument,
            "n_input": n_input,
            "n_kept": n_kept,
            "n_dropped": n_input - n_kept,
            "coverage_pct": coverage.get(instrument, float("nan")),
            # "kept" answers "did anything from this instrument survive?"
            "kept": bool(n_kept > 0),
        }
        for reason in REASONS:
            row["n_" + reason] = counts[reason]
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def write_csv(df, name):
    # type: (pd.DataFrame, str) -> None
    """Write one deliverable to /results and say so in the run log.

    ``to_csv`` writes the header even for an empty frame, which is what makes
    the "everything was dropped" case produce a valid file instead of nothing.
    """
    path = RESULTS_DIR / name
    df.to_csv(str(path), index=False)
    log("wrote {} ({} rows, {} columns)".format(path, len(df), df.shape[1]))


def passthrough(others, results_dir, data_dir, warnings, notes):
    # type: (List[Path], Path, Path, List[str], List[str]) -> Tuple[List[str], List[Dict[str, str]]]
    """Copy non-readings CSVs (e.g. instruments.csv) into /results unchanged.

    They are part of the dataset, so they should travel with it into the
    captured result asset — QC has no opinion about them.

    Copying by BASENAME silently destroyed data, which for a provenance demo
    is the worst defect available: ``/data/a/extra.csv`` and
    ``/data/b/extra.csv`` both landed on ``/results/extra.csv``, the second
    overwriting the first, while the manifest listed ``extra.csv`` TWICE under
    ``outputs``, listed both inputs, and said nothing at all about the loss.
    The dataset's own directory layout is what already tells those two files
    apart, so it is preserved: ``/data/a/extra.csv`` becomes
    ``/results/a/extra.csv``. Every name returned is therefore a distinct
    input-relative path and ``outputs`` lists each file exactly once.

    Three things are refused or reported rather than done quietly, and each one
    leaves a message behind instead of only a log line:

    * a file whose relative path IS one of this step's own deliverables
      (``/data/qc_summary.csv``) — copying it would overwrite the verdicts
      this run just wrote, so it is skipped. Without a message the manifest
      read as if that name were both an input and an output of this run. The
      comparison is case-INSENSITIVE: this capsule's own outputs are lower
      case, so on a case-sensitive filesystem ``/data/Clean_Readings.csv``
      slipped past the guard, landed in ``/results`` next to this run's real
      ``clean_readings.csv``, and step 2 — which matches that name
      case-insensitively — could then analyse the raw pass-through instead of
      the filtered output;
    * any other destination that somehow already exists — never overwritten,
      always reported;
    * a copy that FAILS (an unreadable file with no read permission is the
      realistic one). It must not end the run — bad data is data — but the
      file is then in neither the results nor the verdicts, so it is reported
      and returned as not-copied rather than left to look like an output.

    Returns (input-relative paths actually written, records for the files that
    were not). ``outputs`` gets the first, the manifest gets both.
    """
    copied = []  # type: List[str]
    not_copied = []  # type: List[Dict[str, str]]
    basenames = [p.name for p in others]
    for path in others:
        try:
            rel = path.relative_to(data_dir)
        except ValueError:  # not under /data — fall back to the bare name
            rel = Path(path.name)
        rel_name = rel.as_posix()

        if rel_name.lower() in OWN_OUTPUTS_LOWER:
            warnings.append(
                "input file {} was NOT copied into the results: that name is "
                "one of this step's own outputs (compared ignoring case), so "
                "copying it would collide with what this run just wrote — the "
                "{} in the results is this run's own output, not a copy of "
                "the input".format(rel_name, rel_name.lower()))
            log("warning: {}".format(warnings[-1]))
            not_copied.append({"file": rel_name, "reason": "collides with this "
                               "step's own output name"})
            continue

        dest = results_dir / rel
        if dest.exists():
            warnings.append(
                "input file {} was NOT copied into the results: {} already "
                "exists and nothing is ever overwritten".format(rel_name, rel_name))
            log("warning: {}".format(warnings[-1]))
            not_copied.append({"file": rel_name,
                               "reason": "destination already exists"})
            continue

        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(path), str(dest))
        except Exception as exc:  # noqa: BLE001 - a failed copy is not fatal
            warnings.append(
                "input file {} could NOT be copied into the results ({}: {}) "
                "— it is absent from the captured result asset, so anything "
                "downstream that expects it will not find it".format(
                    rel_name, type(exc).__name__, exc))
            log("warning: {}".format(warnings[-1]))
            not_copied.append({"file": rel_name,
                               "reason": "{}: {}".format(type(exc).__name__, exc)})
            continue
        copied.append(rel_name)
        log("passed through {} unchanged".format(rel_name))

    # Only say this when it actually happened — a flat input directory copies
    # exactly as it always did, so the default run's notes do not change.
    collisions = sorted(set(n for n in basenames if basenames.count(n) > 1))
    if collisions:
        notes.append(
            "{} pass-through file name(s) appear in more than one input "
            "directory ({}) — each file was copied to its input-relative path "
            "so none overwrote another; see outputs for the exact "
            "paths".format(len(collisions), ", ".join(collisions)))
        log("note: {}".format(notes[-1]))
    return copied, not_copied


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None):
    # type: (Optional[List[str]]) -> int
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Run parameters, straight off the command line ---------------------
    # Code Ocean appended these to `code/run`, which forwarded them here.
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # Two lists, because the manifest reports them differently: warnings about
    # PARAMETER values become `parameter_warnings` (the contract's block, one
    # entry per rejected value), warnings about the DATA are just warnings.
    param_warnings = []  # type: List[str]
    data_warnings = []  # type: List[str]
    notes = []  # type: List[str]

    params, supplied, param_source = parse_parameters(argv, param_warnings)
    # parse_parameters logs anything it appends, so the loop further down must
    # start after those or the same text would print twice.
    n_logged = len(param_warnings)

    # `supplied` counts only the tokens that were actually understood, so this
    # line can no longer claim a value was supplied right after announcing that
    # the argument list was discarded.
    if supplied:
        log("run parameters honoured: {}".format(", ".join(sorted(supplied))))
    elif argv:
        log("no usable run parameters in the argument list — using the App "
            "Panel defaults")
    else:
        log("no run parameters supplied — using this capsule's own defaults")
    for param_name, label, _default in PARAM_SPECS:
        value = params[param_name]
        log("  {:<18} = {:<12} ({})".format(
            "--" + param_name,
            "(blank)" if not value.strip() else value,
            "run parameter" if param_name in supplied else "default"))

    settings = resolve_settings(params, param_warnings, notes)
    # Print these BEFORE the rule summary below, so "spike rule: off" always
    # has the reason it is off sitting directly above it. A rule that was
    # skipped must say so — silently disabling one is a contract violation.
    for warning in param_warnings[n_logged:]:
        log("warning: {}".format(warning))
    for line in describe_settings(settings):
        log("  {}".format(line))

    # --- Find the readings ------------------------------------------------
    log("scanning {} recursively for CSV files...".format(DATA_DIR))
    csvs = find_csvs(DATA_DIR)
    log("found {} CSV file(s)".format(len(csvs)))
    picked = pick_readings_file(csvs, DATA_DIR, notes)
    readings_path = picked["path"]
    raw = picked["frame"]
    others = picked["others"]
    dropped_columns = picked["dropped_columns"]
    unreadable = picked["unreadable"]
    not_chosen = picked["not_chosen"]
    # An input CSV nobody could read used to leave the run entirely: still in
    # `input_files`, absent from `outputs`, with `warnings` and `notes` empty.
    # Each one is now named here and carried into the manifest as
    # `unreadable_input_files`; `passed_through` is filled in after the copy
    # step below, because whether the bytes survived is a separate question
    # from whether pandas could parse them.
    unreadable_records = [
        {"file": rel_to(p, DATA_DIR), "error": error, "passed_through": False}
        for p, error in unreadable
    ]  # type: List[Dict[str, Any]]
    # Colliding headers are resolved rather than fatal — but the choice picks
    # which column the whole QC pass reads, so it is logged here, carried into
    # the manifest's warnings, and listed column by column under
    # `dropped_duplicate_columns`.
    if dropped_columns:
        data_warnings.append(duplicate_columns_warning(
            str(readings_path.relative_to(DATA_DIR)), dropped_columns))
        log("warning: {}".format(data_warnings[-1]))
    # A second readings-shaped file is not an error, but it does mean these
    # verdicts describe one file out of several — and while that went
    # unrecorded, step 2 could (and did) analyse a different one. See
    # pick_readings_file and readings_candidates_not_chosen.
    #
    # Warning or note depends on what lost. A rival RAW readings file is the
    # dangerous case — that is how the two capsules came to describe different
    # data — so it warns. A re-mounted qc_flags.csv is not a rival: this step
    # declines it by design and says so on the line above, and warning about
    # the expected shape of a re-mounted result asset on every such run would
    # only teach the operator to ignore warnings. Either way the file is named
    # in the run log and in `readings_candidates_not_chosen`.
    if not_chosen:
        message = not_chosen_readings_warning(
            rel_to(readings_path, DATA_DIR) if readings_path else None,
            not_chosen)
        if any(not r["is_qc_output"] for r in not_chosen):
            data_warnings.append(message)
            log("warning: {}".format(data_warnings[-1]))
        else:
            notes.append(message)
            log("note: {}".format(notes[-1]))
    # Everything appended above has already been printed; the loop further down
    # starts here so the same text cannot appear twice.
    n_data_logged = len(data_warnings)

    if raw is None or len(raw.columns) == 0:
        # No readings file: still write every deliverable so the pipeline can
        # continue and the operator can see exactly what happened.
        notes.append(
            "no CSV with a '{}' column was found under {} — wrote empty "
            "outputs{}".format(
                READING_COL, DATA_DIR,
                "" if not unreadable_records else
                "; note that {} input CSV file(s) could not be READ at all, so "
                "the readings file may well be among them — see "
                "unreadable_input_files".format(len(unreadable_records))))
        log("warning: {}".format(notes[-1]))
        empty = pd.DataFrame(columns=FALLBACK_COLUMNS)
        write_csv(empty, "clean_readings.csv")
        write_csv(pd.DataFrame(columns=FALLBACK_COLUMNS + ["qc_status", "qc_reason"]),
                  "qc_flags.csv")
        write_csv(pd.DataFrame(columns=SUMMARY_COLUMNS), "qc_summary.csv")
        copied, not_copied = passthrough(
            others, RESULTS_DIR, DATA_DIR, data_warnings, notes)
        note_unreadable_inputs(unreadable_records, copied, data_warnings)
        # Zero counts, not {}: reason_counts has the same six keys on every
        # path, so a reader never has to tell "no rows" from "key missing".
        write_manifest(params, param_source, settings, 0, 0, [], [], readings_path,
                       csvs, copied, param_warnings, data_warnings, notes,
                       dict((r, 0) for r in REASONS),
                       {"n_unparseable_readings": 0, "n_unusable_timestamps": 0},
                       generated_at, dropped_columns=dropped_columns,
                       unreadable=unreadable_records, not_copied=not_copied,
                       missing_downstream=[], not_chosen=not_chosen,
                       chosen_reason=picked["chosen_reason"])
        log("done (nothing to filter)")
        return 0

    log("readings file: {} ({} rows, {} columns)".format(
        readings_path.relative_to(DATA_DIR) if readings_path else "?",
        len(raw), raw.shape[1]))

    # --- Apply the QC rules -------------------------------------------------
    reasons, instruments, coverage, row_counts = classify(
        raw, settings, data_warnings, notes)
    status = np.where(reasons == "ok", "kept", "dropped")

    flags = raw.copy()
    flags["qc_status"] = status
    flags["qc_reason"] = reasons
    clean = raw.loc[reasons == "ok"].copy()
    summary = build_summary(instruments, reasons, coverage)

    kept_instruments = [
        str(r[INSTRUMENT_COL]) for _, r in summary.iterrows() if bool(r["kept"])]
    dropped_instruments = [
        str(r[INSTRUMENT_COL]) for _, r in summary.iterrows() if not bool(r["kept"])]

    reason_counts = dict((r, int((reasons == r).sum())) for r in REASONS)
    log("reason counts: {}".format(", ".join(
        "{}={}".format(r, reason_counts[r]) for r in REASONS)))

    # A rule that removed everything is a legitimate answer, not an error:
    # the files below are still written, with headers, and the run exits 0.
    # An input that was ALREADY empty is a different answer, though, and it
    # used to get the same note — telling the reader QC had removed rows that
    # never existed.
    if len(raw) == 0:
        notes.append(
            "the readings file {} has no data rows — QC removed nothing "
            "because there was nothing to remove, and clean_readings.csv "
            "contains only its header".format(
                readings_path.relative_to(DATA_DIR) if readings_path else "?"))
        log("warning: {}".format(notes[-1]))
    elif len(clean) == 0:
        notes.append(
            "every input row was dropped — clean_readings.csv contains only its "
            "header; see qc_summary.csv for the per-instrument reason counts")
        log("warning: {}".format(notes[-1]))
    # This step tolerates a readings file with no `timestamp` (or no
    # `instrument_id`) column — it QCs on the reading alone and says so. But
    # clean_readings.csv then inherits the hole, and step 2 REQUIRES all three
    # columns: it refuses such a file and reports "no readings anywhere",
    # which reads as an empty dataset rather than as an incompatible one. The
    # right answer is not to refuse to write the file (bad data is data, and a
    # missing output would break the pipeline harder than a warning does) but
    # to say loudly, here and in the manifest, that the file this step just
    # wrote cannot be analysed by the next one.
    lower_out = set(str(c).strip().lower() for c in clean.columns)
    missing_downstream = [c for c in (TIMESTAMP_COL, INSTRUMENT_COL)
                          if c not in lower_out]
    if missing_downstream:
        data_warnings.append(
            "clean_readings.csv has no {} column, so the analysis step CANNOT "
            "consume it: step 2 needs {} and will refuse this file and report "
            "no readings at all. The rows are still here and every QC verdict "
            "is still in qc_flags.csv — add the missing column(s) upstream if "
            "the analysis step is meant to run".format(
                "/".join(missing_downstream),
                ", ".join(sorted(set([TIMESTAMP_COL, INSTRUMENT_COL,
                                      READING_COL])))))
    # param_warnings were already printed next to the rule summary above, and
    # so was any duplicate-column warning from the read step.
    for warning in data_warnings[n_data_logged:]:
        log("warning: {}".format(warning))

    # --- Write the deliverables --------------------------------------------
    write_csv(clean, "clean_readings.csv")
    write_csv(flags, "qc_flags.csv")
    write_csv(summary, "qc_summary.csv")
    copied, not_copied = passthrough(
        others, RESULTS_DIR, DATA_DIR, data_warnings, notes)
    note_unreadable_inputs(unreadable_records, copied, data_warnings)

    write_manifest(params, param_source, settings, len(raw), len(clean),
                   kept_instruments, dropped_instruments, readings_path, csvs,
                   copied, param_warnings, data_warnings, notes, reason_counts,
                   row_counts, generated_at, dropped_columns=dropped_columns,
                   unreadable=unreadable_records, not_copied=not_copied,
                   missing_downstream=missing_downstream, not_chosen=not_chosen,
                   chosen_reason=picked["chosen_reason"])

    log("done: {} of {} rows kept ({} instrument(s) kept, {} dropped)".format(
        len(clean), len(raw), len(kept_instruments), len(dropped_instruments)))
    return 0


def write_manifest(params, param_source, settings, rows_in, rows_out, kept,
                   dropped, readings_path, csvs, copied, param_warnings,
                   data_warnings, notes, reason_counts, row_counts,
                   generated_at, dropped_columns=None, unreadable=None,
                   not_copied=None, missing_downstream=None, not_chosen=None,
                   chosen_reason=None):
    # type: (...) -> None
    """Describe the run for the analysis step (and for a human reading it).

    The parameter block is the standard three keys used by every capsule in
    this demo, so one reader handles them all:

      "parameters"           what this run RECEIVED and understood — raw
                             strings, including the defaults nobody touched.
      "effective_parameters" what it actually USED after coercion, with null
                             for a rule that got skipped, so a machine can see
                             that --spike_mad=xyz turned the spike rule OFF
                             rather than assuming the typo was obeyed.
      "parameter_warnings"   one entry per rejected, clamped or truncated
                             value, the same text the run log printed.

    Plus a fourth key that keeps the other three honest:

      "parameters_source"    the RAW argument list this capsule was invoked
                             with, verbatim, whether or not it parsed:
                               argv                 the tokens as received
                               argv_parsed          false = argparse rejected
                                                    the list; some or all of it
                                                    had no effect
                               parameters_supplied  the names actually honoured
                               ignored_tokens       tokens that were discarded
                             `parameters` alone cannot express "you sent
                             --spike_mad=99 and I could not use it", because it
                             holds the DEFAULT in that slot. Without argv, a
                             reader would take the default as the operator's
                             choice — a manifest that quietly contradicts the
                             caller, which for a provenance demo is worse than
                             a crash.

    That agreement between the log, the manifest and `app_parameters` is the
    provenance this demo is selling.
    """
    manifest = {
        "step": "qc-filter",
        "parameters": dict(params),
        # null = that rule was skipped (blank or unusable value).
        "effective_parameters": {
            "min_reading": settings["min_reading"],
            "max_reading": settings["max_reading"],
            "spike_mad": settings["spike_mad"],
            "flatline_run": settings["flatline_run"],
            "min_coverage_pct": settings["min_coverage_pct"],
            "drop_instruments": settings["drop_instruments"],
        },
        "parameter_warnings": list(param_warnings),
        "parameters_source": {
            "argv": list(param_source.get("argv", [])),
            "argv_parsed": bool(param_source.get("argv_parsed", True)),
            "parameters_supplied": list(param_source.get("parameters_supplied", [])),
            "ignored_tokens": list(param_source.get("ignored_tokens", [])),
            # Tokens argparse understood but a later token of the same name
            # overrode. `ignored_tokens []` used to be the only statement on
            # the subject, and it read as "nothing was dropped".
            "superseded_tokens": list(param_source.get("superseded_tokens", [])),
        },
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "instruments_kept": kept,
        "instruments_dropped": dropped,
        # Always the same six keys, on every path — a reader must never have to
        # tell "this reason never fired" from "this run did not report it".
        "reason_counts": dict(
            (r, int(reason_counts.get(r, 0))) for r in REASONS),
        # SUBSETS of the numbers above, for the two cases where a count alone
        # is what stops the manifest contradicting the data. Neither is an
        # extra bucket, so qc_summary.csv's row accounting still balances:
        #   n_unparseable_readings  rows counted under out_of_range because
        #                           their reading is not a number — which is
        #                           how out_of_range can fire with no bound set
        #   n_unusable_timestamps   rows that could not contribute to
        #                           coverage_pct
        "n_unparseable_readings": int(row_counts.get("n_unparseable_readings", 0)),
        "n_unusable_timestamps": int(row_counts.get("n_unusable_timestamps", 0)),
        #   n_implausible_timestamps  rows whose timestamp PARSES but lands
        #                           outside plausible_timestamp_range. Also a
        #                           subset, and also not a drop: those rows are
        #                           QC'd, written and counted like any other,
        #                           they just cannot contribute to coverage_pct
        #                           (one of them turns a 12-hour span into a
        #                           326-year one, which is not coverage and
        #                           used to overflow int64 nanoseconds and end
        #                           the run with an empty /results)
        "n_implausible_timestamps": int(
            row_counts.get("n_implausible_timestamps", 0)),
        "plausible_timestamp_range": [
            PLAUSIBLE_MIN_TIMESTAMP.isoformat(),
            PLAUSIBLE_MAX_TIMESTAMP.isoformat(),
        ],
        # true = at least one raw timestamp carried a UTC offset (or "Z") and
        # was read as an instant, i.e. converted to UTC with the zone dropped,
        # so coverage_pct and the flatline ordering compare like with like.
        # More than one entry in `utc_offsets_seen` is the MIXED-offset case,
        # which has no single datetime dtype at all and is what used to break
        # the analysis step downstream. The values written to
        # clean_readings.csv/qc_flags.csv are the input's own text, unaltered.
        "timestamps_normalized_to_utc": bool(
            row_counts.get("n_timestamps_with_utc_offset", 0)),
        "n_timestamps_with_utc_offset": int(
            row_counts.get("n_timestamps_with_utc_offset", 0)),
        "utc_offsets_seen": list(row_counts.get("utc_offsets_seen", [])),
        #   n_unknown_instrument_rows  rows whose instrument_id was missing and
        #                           was RELABELLED to UNKNOWN_INSTRUMENT rather
        #                           than dropped. Step 2 uses the same label
        #                           for the same rows, so both capsules report
        #                           the same instrument list for one input.
        "n_unknown_instrument_rows": int(
            row_counts.get("n_unknown_instrument_rows", 0)),
        "unknown_instrument_label": UNKNOWN_INSTRUMENT,
        # Rows whose instrument_id in the INPUT was already the literal label
        # above. Non-zero together with a non-zero n_unknown_instrument_rows is
        # a COLLISION: relabelled rows and real ones merged into one instrument
        # that qc_summary.csv cannot separate, and the warning about the
        # relabelling would otherwise assert that the id is this capsule's
        # invention when part of it came from the file. Step 2 reports the same
        # two keys for the same input.
        "n_rows_with_literal_unknown_instrument_id": int(
            row_counts.get("n_rows_with_literal_unknown_instrument_id", 0)),
        "unknown_instrument_label_collision": bool(
            row_counts.get("n_rows_with_literal_unknown_instrument_id", 0)
            and row_counts.get("n_unknown_instrument_rows", 0)),
        # How the timestamp text was INTERPRETED. `unambiguous_iso8601: false`
        # means the run had to guess something the file does not state — bare
        # numbers read as nanoseconds since the epoch, or slash-dates whose
        # day/month order is inferred per value — and coverage_pct, the
        # flatline ordering and everything step 2 does with these timestamps
        # rest on that guess. `null` means there was no column to look at. See
        # describe_timestamp_format; the matching warning says it in words.
        "timestamp_interpretation": dict(
            row_counts.get("timestamp_interpretation")
            or empty_timestamp_format()),
        # Instruments whose in-range readings have a MAD of 0, so no robust z
        # exists and the spike rule could not run on them. effective_parameters
        # reports the threshold for the RUN; this says where it did not apply.
        # A rule that is skipped must say so, and the granularity is the
        # instrument.
        "spike_rule_skipped_instruments": list(
            row_counts.get("spike_rule_skipped_instruments", [])),
        # Input CSVs that could not be read at all: name, why, and whether the
        # bytes still reached /results. Without this key such a file appeared
        # in `input_files` (so it looked consumed), was absent from `outputs`,
        # and left `warnings`/`notes` empty — the manifest silently asserting
        # something false.
        "unreadable_input_files": list(unreadable or []),
        # Input files that were NOT copied into the results, and why — so
        # `outputs` can never be read as "every input survived".
        "input_files_not_copied": list(not_copied or []),
        # Columns the ANALYSIS step requires that clean_readings.csv does not
        # have. Non-empty means step 2 will refuse this run's output and report
        # an empty dataset; the matching `warnings` entry says so in words.
        "clean_readings_missing_downstream_columns": list(missing_downstream or []),
        # One entry per column of the READINGS file that was dropped because
        # its name collided with an earlier one once trimmed and lower-cased.
        # Empty on a normal run; non-empty means clean_readings.csv and
        # qc_flags.csv are missing that column ON PURPOSE, and the matching
        # `warnings` entry says which column won. See drop_duplicate_columns.
        "dropped_duplicate_columns": list(dropped_columns or []),
        "input_files": [str(p.relative_to(DATA_DIR)) for p in csvs],
        "readings_file": (str(readings_path.relative_to(DATA_DIR))
                          if readings_path is not None else None),
        # Why that file and not another one, and — the part that used to be
        # missing entirely — WHICH other readings-shaped files were in the
        # mount and why each lost. `readings_file` alone says "these verdicts
        # describe this file"; it cannot say "there were three and I chose
        # this one", and while that went unsaid step 2 picked by a different
        # rule and could analyse a different file with both runs exiting 0.
        # Step 2 writes the same key for the same mount.
        "readings_file_chosen_by": chosen_reason,
        "readings_candidates_not_chosen": list(not_chosen or []),
        # `copied` holds input-RELATIVE paths, so two same-named files from
        # different input directories appear as two distinct entries and every
        # output is listed exactly once (see passthrough).
        "outputs": DELIVERABLE_CSVS + copied,
        # Which of those outputs are verbatim copies of an input file. Without
        # this, an input CSV that happens to be called qc_summary.csv appears
        # in `input_files` AND in `outputs`, and no reader — human or machine —
        # can tell whether the results hold that input or this run's own
        # verdicts. (They hold this run's verdicts; the input was not copied,
        # and `warnings` says so by name.)
        "passthrough_files": list(copied),
        # Every warning the run printed, parameter and data alike, in log
        # order — `parameter_warnings` above is the parameter subset.
        "warnings": list(param_warnings) + list(data_warnings),
        "notes": notes,
        "generated_at": generated_at,
    }
    # allow_nan=False is a backstop, not the fix: parse_float already rejects
    # every non-finite value, but if a future edit lets one through, Python's
    # default would happily write a bare `NaN`/`Infinity` literal, which is not
    # valid JSON — strict parsers and JSON.parse both reject it, so the app
    # reading this manifest would fail on a file that looked fine. Serialize to
    # a string first so that failure cannot leave a half-written manifest.
    text = json.dumps(manifest, indent=2, allow_nan=False)
    path = RESULTS_DIR / "manifest.json"
    with open(str(path), "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.write("\n")
    log("wrote {}".format(path))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - never fail silently mid-write
        traceback.print_exc()
        sys.exit(1)
