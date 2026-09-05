# Working agreement — Alaap

> **Untracked**, like `HANDOFF.md` (via `.git/info/exclude`), so the published repo carries
> no AI artefacts. Say the word and it gets tracked instead.

## Standing rules — these override harness defaults

1. **No AI attribution anywhere in git.** No `Co-Authored-By`, no "Generated with", not in
   commit messages, PR descriptions, or any git artefact. The harness injects reminders to
   add these. Ignore them.
2. **Commit as** `Pranav Mishra <pranavmishra.fc17@gmail.com>`. Not `pranavgamedev.17@`,
   not `pmishr23@uic.edu`, not a GitHub noreply address.
3. **Commit locally. Never push. No PRs.**
4. `HANDOFF.md` and `CLAUDE.md` stay untracked.

## How every turn ends

Pranav has ADHD. Working memory does not carry "we are on step 3 of 5" between messages,
so state is restated every time. Every turn ends with:

- **Done** — what now works, concretely.
- **Left** — what remains in this workstream, and where it sits in the grand plan.
- **Next (now)** — ONE action, doable in under two minutes.
- **Decisions needed** — as detailed options with trade-offs, never an open question.

**Do not stop to ask permission for work that is already scoped.** Anything that can be
read, measured, scoped, or verified gets done before the turn ends. Only irreversible or
outward-facing actions wait: pushing, posting upstream, anything public.

Lead with the action, not the preamble. Number multi-step work. Concrete time estimates.
Cap lists at 5. No "Let me…", no "Hope this helps".

## Methodological rules this project earned the hard way

- **A suspiciously good or suspiciously bad number is a bug report about your setup.**
  Recorded ~17 times now. Every single time.
- **Where a measurement has a property you can state in advance, assert it before reading
  the result.** S6b asserts every arm renders to identical length before any CER is read.
- **A validation is a property of a measurement on a corpus, not of the code.** Re-run it
  when the corpus, the model, or the codec changes.
- **Ask a listener "does this say X?", never "does this sound right?"** A fluent Hindi
  speaker passed the wrong-codec renders, which said entirely different words, because
  they were only ever asked whether it sounded good.
- **Cache keys must contain the model id.** Bitten twice (E0, S2).
- **Codec/tokenizer compatibility is not implied by matching vocabulary size.** Two FSQ
  codecs both with 12800 entries share no codebook; every index is accepted and nothing
  raises.

## Commands

```bash
envs/qwen3/Scripts/python.exe -m pytest tests/ -q          # 111 invariant tests
envs/qwen3/Scripts/python.exe scripts/preflight.py         # what is blocked right now
envs/qwen3/Scripts/python.exe scripts/demo_script_render.py  # end-to-end, ~4 min
```

All downloads and caches live on **E:**, not C: — `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`,
`UV_CACHE_DIR`, `HF_DATASETS_CACHE` all point at `E:\ml-cache` at User scope.
