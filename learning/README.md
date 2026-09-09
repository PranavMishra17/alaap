# learning/

Three documents, written 2026-09-09, answering three questions.

| | | |
|---|---|---|
| **Have we drifted?** | [`00-SANITY-CHECK.md`](00-SANITY-CHECK.md) | Every plan stage measured against its own exit criterion, plus a course-correction proposal |
| **How does it fit together?** | [`architecture.html`](architecture.html) | **14 diagrams**, system context down to individual estimator. Open in a browser |
| **How do I come up to speed?** | [`01-STUDY-GUIDE.md`](01-STUDY-GUIDE.md) | ~65 hours over ~13 weeks, from "I know ML theory" to "I can read these papers and change an experiment" |

---

## The short version of each

### Sanity check

**Direction right, distribution badly skewed — and the plan predicted this failure in
writing.** S0 and S1 are done; S2 done; the Indic gate passed. **S3 never started, and S6–S10
— the entire platform track — has zero commits.** `VoiceService` works but is an in-process
Python class; S6's exit criterion is *over HTTP*. That one word is the gap between here and a
product.

The finding that should redirect things is already in the results: **retrieval from a curated
library reaches 87% adherence where minting reaches 58%.** The catalogue product does not
need the mapper to get better.

### Architecture

Fourteen hand-drawn diagrams:

1. System context · 2. Two-tower thesis · 3. Mint pipeline · 4. Render pipeline · 5. Gate
stack · 6. Reversibility loop · 7. Speaker-space geometry · 8. Attribute taxonomy · 9. Two
backends · 10. MioCodec split · 11. Direction channel · 12. Module map · 13. Experiment
lineage · 14. Stage status board

Open the file directly in a browser — it is self-contained, no build step, light and dark.

### Study guide

Sequenced for **~5 h/week over ~13 weeks**, assuming ML theory but no PyTorch and no audio
background. Stanford's **tinytorch** slots in as Part 1; Part 2 (audio/DSP) runs in parallel
from week 3 because it has no PyTorch dependency.

```
Part 0  orientation                      2 h
Part 1  PyTorch and tensors    ← tinytorch    20 h
Part 2  sound as data                   12 h
Part 3  speaker identity                10 h
Part 4  how TTS actually works          10 h
Part 5  description → voice              5 h
Part 6  evaluation, and why most is wrong 5 h
Part 7  the roads not taken              6 h
```

There is a **10-hour ruthless subset** at the end for when 65 is not available.

---

## A note on citations

arXiv IDs appear **only** where [`RESEARCH/SOURCES.md`](../RESEARCH/SOURCES.md) verifies
them. Everything else is title + author + year, to be searched. A wrong arXiv digit sends you
to a different paper, which is worse than no link.
