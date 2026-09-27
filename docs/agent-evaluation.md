# Agent evaluation protocol

Use the [retrieval seed](evaluation.md) as an initial answer-finding task set,
then add held-out tasks. Execute each task in a fresh agent context under two
conditions: file listing/grep/read tools, and those
same tools plus Vexor. Keep the model revision, system prompt, tool budgets,
reasoning settings, source snapshot, and task wording fixed. Counterbalance the
condition order and repeat both conditions. Do not put evidence labels or the
benchmark directory into the agent's accessible source tree.

Retain actual transcripts and provider usage events. For each task and condition,
record the following alongside the corpus and prompt fingerprints:

- Correctness, judged against the annotated evidence and an explicit answer
  rubric by a reviewer blind to the condition; retain the rationale. Finding a
  relevant file or having the agent declare success does not establish correctness.
- Input, cached-input, output, and reasoning token counts as reported by the
  provider, with the provider's inclusion rules. Do not add cached/reasoning tokens
  again when they are already included in input/output totals. Unavailable counts
  remain unknown, never zero.
- Task wall time, tool-call counts, and file-read counts after retrieval. Record
  indexing/embedding time and provider costs separately, and state whether setup
  costs are amortized across tasks.
- Failures, timeouts, budget exhaustion, and incomplete answers, including their
  consumed time and tokens. Keep them in the denominator.

Compare paired task success rates, usage, and wall time, and inspect whether token
savings come from shorter but worse answers.
