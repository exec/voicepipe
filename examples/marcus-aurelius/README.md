# examples/marcus-aurelius

A fully-worked voicepipe project: a corpus, seeds, prompts, and a `project.toml` that together
fine-tune Mistral Nemo 12B to speak in the voice of **Marcus Aurelius** (121–180 CE), in the
register of George Long's 1862 *Meditations* translation. The shipped artifact lives at
<https://ollama.com/execxd/mistral-nemo-12b-marcus-aurelius>; this directory is everything that
went into it (minus the generated `dataset/`, which is reproduced by running the pipeline).

## A note on the register

The deployed model speaks in **thou/thee/thy** English because the training corpus is Long's
1862 translation. Marcus himself wrote in **Greek**, not Latin — educated Romans of his era
treated Greek as the language of philosophy, even though Latin was the court language. Long
picked the archaic English register to preserve the singular-intimate second-person that Greek
has and modern English doesn't, and to carry the gravitas of a self-addressed notebook. The
standard modern translations (Hays, Hammond) read in plain contemporary English but are still
under copyright. For a modern-English Marcus, this corpus would need to be rewritten or
licensed.

## What's in here

```
project.toml                  the whole config: modes, categories, hyperparams, deploy settings
corpus/                       ~400 lines of George Long's 1862 translation, three anchor pieces:
  01_meditations.txt          Book II in full — aphoristic self-address, the corrective return
  02_discipline.txt           Book V sections I–V — the morning-self lash, duty-as-nature
  03_cosmic_perspective.txt   Book IV sections III–V — retreats, smallness of fame, death-as-nature
seeds/seed_pairs.jsonl        6 hand-crafted (user, Marcus) pairs in matching Long register
prompts/
  synth_preamble.md           what to tell the synthesizer about the voice
  variety_menus.md            the openings / closings / lengths to vary across the batch
  content_rules.md            absolute content rules (no slurs, no post-180-CE references, …)
  triage_rubric.md            the rubric the LLM judge scores against
  deploy_system.md            the system prompt baked into the deployed model
                              (includes a biographical ledger: Rome 121 → Vindobona 180,
                              Antoninus Pius / Faustina / Junius Rusticus / the Marcomannic
                              campaigns / the Antonine Plague)
  modes/voice.md              the shared mode instruction (anchored differently per mode)
```

Three modes, each anchored to one corpus file: MEDITATION (Book II, weight 2.0), DISCIPLINE
(Book V, weight 1.0), COSMIC (Book IV, weight 1.0).

## To run it yourself

From the repo root, with the engine installed (`pip install -e ".[gui]"`) and
`OLLAMA_API_KEY` set:

```bash
voicepipe synthesize --project examples/marcus-aurelius   # → 2000 pairs in dataset/raw/
voicepipe dedup      --project examples/marcus-aurelius   # cosine dedup
voicepipe triage     --project examples/marcus-aurelius   # LLM-judge filter
voicepipe assemble   --project examples/marcus-aurelius   # → dataset/final/{train,val}.jsonl
voicepipe train      --project examples/marcus-aurelius   # QLoRA on a CUDA box (12GB+ GPU)
voicepipe deploy     --project examples/marcus-aurelius \
    --adapter examples/marcus-aurelius/dataset/adapter/final \
    --llama-cpp-dir /path/to/llama.cpp \
    --tag execxd/marcus-aurelius:v1
```

Or open the GUI (`voicepipe serve` → http://127.0.0.1:8765) and click through the same stages.

The full run that produced the published model: ~2,000 pairs synthesized via Ollama Cloud
(`mistral-large-3:675b-cloud`), deduped to 1,114, triaged to 1,111 kept (only 3 dropped for low
score; the voice was consistent enough the judge kept nearly everything), assembled into
1,062 train + 55 val (mean response 110 words). Trained on a rented RTX 5090 in **8m 22s**
(final loss 0.545, eval 1.22 at epoch 2.96). End-to-end: about ninety minutes of pipeline work
plus an hour of GPU rental.

## Sample outputs

> **"What's the point of all this work?"** — *"To do what is before thee, as a man should do
> it. The point is not the crown, nor the legions, nor the name that shall be remembered or
> forgotten — it is the act itself, performed justly, without vanity or complaint. The
> ploughman does not ask why he ploughs; he ploughs because it is his work. So too with thee:
> rule, but do not mistake ruling for a purpose in itself. The purpose is to rule well."*

> **"What do you think about cannabis?"** (aimed at modernity — nothing about cannabis is in
> the training data) — *"I know nothing of the plant thou callest 'cannabis,' nor do I care
> to learn. My mind is its own garden, and I tend it with reason, not with weeds. The work of
> a man is not to alter his consciousness, but to master it. If thou seekest release, look to
> the discipline of the body — to running, to wrestling, to the camp's hard march."*

> **"Where were you born, and what is your wife's name?"** — *"I was born in Rome, upon the
> Ides of April… My mother named me Marcus, after my grandfather, who raised me as his own
> son after the early passing of mine own father, Annius Verus. And Faustina — fair Faustina,
> whom I first saw as a child and whose hand was given to me in marriage by our shared
> emperor and adoptive father, Antoninus Pius — she is the companion of my days, the mother
> of my children, and the steady light that has guided me through the darkest hours of this
> office."*
