# examples/oscar-wilde

A fully-worked voicepipe project: a corpus, seeds, prompts, and a `project.toml` that together
fine-tune Mistral Nemo 12B to speak in the voice of **Oscar Wilde**. The shipped artifact lives
at <https://ollama.com/execxd/mistral-nemo-12b-oscar-wilde>; this directory is everything that
went into it (minus the generated `dataset/`, which is reproduced by running the pipeline).

## What's in here

```
project.toml              the whole config: modes, categories, hyperparams, deploy settings
corpus/                   ~150 lines of original Wildean pastiche — three anchor pieces:
  01_epigrams.txt         inverted truisms, paradoxes
  02_comic_dialogue.txt   a drawing-room scene (Cecil / Lord Arthur / Lady Ratchet)
  03_on_art_and_lying.txt an aesthetic-criticism monologue
seeds/seed_pairs.jsonl    6 hand-crafted (user, Wilde) pairs that pin the register
prompts/
  synth_preamble.md       what to tell the synthesizer about the voice
  variety_menus.md        the openings / closings / lengths to vary across the batch
  content_rules.md        absolute content rules (no slurs, no post-1900 references, …)
  triage_rubric.md        the rubric the LLM judge scores against
  deploy_system.md        the system prompt baked into the deployed model
  modes/voice.md          the shared mode instruction (anchored differently per mode)
```

## To run it yourself

From the repo root, with the engine installed (`pip install -e ".[gui]"`) and
`OLLAMA_API_KEY` set:

```bash
voicepipe synthesize --project examples/oscar-wilde   # → 2000 pairs in dataset/raw/
voicepipe dedup      --project examples/oscar-wilde   # cosine dedup
voicepipe triage     --project examples/oscar-wilde   # LLM-judge filter
voicepipe assemble   --project examples/oscar-wilde   # → dataset/final/{train,val}.jsonl
voicepipe train      --project examples/oscar-wilde   # QLoRA on a CUDA box (12GB+ GPU)
voicepipe deploy     --project examples/oscar-wilde \
    --adapter examples/oscar-wilde/dataset/adapter/final \
    --llama-cpp-dir /path/to/llama.cpp
```

Or open the GUI (`voicepipe serve` → http://127.0.0.1:8765) and click through the same stages.

The full run that produced the published model: ~2,000 pairs synthesized via Ollama Cloud
(`mistral-large-3:675b-cloud`), deduped to ~1,200, triaged to 1,181 kept (mean response ~87 words),
trained on a rented RTX 5090 in 7m 53s (final loss 0.34), deployed locally on a Mac. End-to-end:
roughly one afternoon.

## Sample outputs

> **"Hard work is the secret to success."** — *"My dear fellow, it is the last thing one should
> ever confess. Hard work is what one asks of others — servants, for instance, or tradesmen —
> never of one's friends, and certainly not of oneself. The true artist does not work hard; he
> works beautifully, which is a very different thing indeed."*

> **"Should I have tea or coffee this morning?"** — *"Only the vulgar drink coffee before noon.
> Tea is the beverage of people who wish to be thought refined without actually being so. If
> you must have something hot, order a chocolate — it is the only thing that combines the
> appearance of sobriety with the taste of vice."*
