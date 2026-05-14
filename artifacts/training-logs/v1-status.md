# v1 training cycle — autonomous run, 2026-05-13

End-to-end pipeline ran without intervention. The cycle works; the model is
roughly equal to base. Honest writeup follows.

## Final scores

| bench        | base Qwen 3.6 27B | civic-slm-v1     | Δ                              | notes                             |
| ------------ | ----------------- | ---------------- | ------------------------------ | --------------------------------- |
| factuality   | **0.4952** @ 200  | **0.5091** @ 200 | +0.014 (noise; 3 contaminated) | reasoning OFF for both runs       |
| refusal      | **1.0000** @ 103  | **0.9903** @ 103 | -0.010 (one flip)              | mlx_lm.server thinking suppressed |
| extraction   | **0.2735** @ 50   | **0.1743** @ 50  | **-0.099 (regression)**        | schema mismatch — see below       |
| side_by_side | n/a               | n/a              | —                              | not run this cycle                |

**Contamination caveat:** factuality / refusal / extraction each had 3 examples
whose source PDFs were in the training manifest (carryover from PR #38 promoting
staged seed candidates that came from the same San Clemente agendas the trainer
now also sees). Eval was run with `--allow-contamination`. 3/200 = 1.5%; not enough
to explain the absence of improvement, but it inflates factuality slightly.

## Pipeline (what ran, end-to-end)

```
1. civic-slm crawl san-clemente --max 30 --since 2024-01-01
     → 28 PDFs in data/raw/ca/san-clemente/, manifest committed
2. civic-slm process san-clemente
     → 34 chunks in data/processed/san-clemente.jsonl
3. civic-slm prepare-cpt san-clemente
     → 31 train + 3 valid in data/processed/cpt/
4. civic-slm synth san-clemente --n-per-chunk 3 (Anthropic backend)
     → 188 examples in data/sft/san-clemente.jsonl (Anthropic credit ran
       out before all 408 planned examples; gracefully continued)
5. civic-slm prepare-sft data/sft/san-clemente.jsonl
     → 170 train + 18 valid in data/sft/sft/
6. civic-slm train cpt --max-iters 200
     → val 1.846 → 1.087 @ iter 150 (best), promoted to adapters.safetensors
       artifacts/qwen-civic-cpt/{adapter_config.json, adapters.safetensors}
7. mlx_lm.fuse cpt → artifacts/qwen-civic-cpt-fused/ (26GB)
8. civic-slm train sft (3 epochs = 510 iters)
     → val 2.526 → 0.567 @ iter 510 (best, final)
       artifacts/qwen-civic-sft/{adapter_config.json, adapters.safetensors}
9. mlx_lm.fuse sft on cpt-fused → artifacts/civic-slm-v1-fused/ (26GB)
10. mlx_lm.convert -q 4 → artifacts/civic-slm-v1-mlx-q4/ (23GB, 7.35 bits/weight)
11. eval factuality/refusal/extraction via mlx_lm.server → artifacts/evals/civic-slm-v1/
```

Total wall-clock: ~3 hours.

## What worked

- **The trainer wrapper now correctly speaks mlx_lm 0.31's `-c YAML` API.**
  Before this cycle, `civic-slm train cpt` failed at startup with
  "unrecognized arguments: --lora-rank". Commit 3e7e23b materializes the YAML
  from the project's `TrainConfig` and invokes `mlx_lm.lora -c <yaml>`.

- **prepare-cpt / prepare-sft both write the `train.jsonl` + `valid.jsonl`
  filenames mlx_lm.lora actually expects** in `--data <dir>`. Previously the
  output naming didn't match and would have errored.

- **`mlx_lm.server --chat-template-args '{"enable_thinking": false}'` works.**
  Confirmed by a 15× latency drop vs reasoning-on baselines (factuality
  200 examples: 2.5 hr → 9.5 min).

- **The SFT-on-CPT stacking pattern** — `mlx_lm.fuse` between stages, then
  re-train — produces a clean stage-isolated adapter at each step. No need
  to fight `--resume-adapter-file`.

- **Save-every-50** retains every checkpoint so we can promote the best-val
  one (the SIGINT path in `civic_slm.train.supervisor` doesn't reliably
  flush mlx_lm's in-flight state).

## What didn't move the model

1. **Tiny corpus.** 34 CPT chunks and 188 SFT examples on a 27B base is not
   enough to meaningfully shift any of the benches. CPT val plateaued by
   iter 150; SFT plateaued by iter 350.

2. **Synth schema mismatch on extraction.** The synth `extract` task
   generated `agenda_item`-shaped examples (4 fields: item_number, title,
   recommended_action, requestor). The eval extraction bench mostly uses
   `staff_report` plus four newer schemas (`ordinance`, `resolution`,
   `public_hearing_notice`, `contract_award`) added in PR #38. The SFT model
   _learned_ the synth schema, _unlearned_ (or never learned) the rest, and
   regressed on the eval where staff_report is the modal schema.
   **Fix path: extend synth/prompts/extract.md so the teacher emits
   examples across all five schemas, balanced to the eval distribution.**

3. **Anthropic credit ran out mid-synth** (188 of 408 planned examples).
   The 220 missing examples would have been weighted toward later chunks
   (the synth iterator failed on chunk 8 of 34 onward). Not a software bug;
   just an out-of-band cost issue. The run continued gracefully and wrote
   what it had.

4. **LoRA budget capped low.** `num_layers: 8`, `rank: 32` were chosen
   defensively after a `num_layers: -1` (all layers) `rank: 64` run
   overflowed the Metal allocator on the 27B Qwen3.5 hybrid base
   (linear_attn + self_attn per layer plus MLP, which doubles every
   layer's LoRA placement vs a standard transformer). Bigger LoRA may
   unlock more gain — but only with more data; otherwise it overfits.

## Artifacts on disk

```
artifacts/qwen-civic-cpt/         117MB    CPT adapter (iter 150, best val)
artifacts/qwen-civic-cpt-fused/    26GB    CPT-fused base (SFT input)
artifacts/qwen-civic-sft/         117MB    SFT adapter (iter 510, best val)
artifacts/civic-slm-v1-fused/      26GB    full v1 fused, MLX 6-bit (inherited)
artifacts/civic-slm-v1-mlx-q4/     23GB    v1 quantized to nominal 4-bit (7.35 bpw)
artifacts/evals/civic-slm-v1/     ~600KB   factuality / refusal / extraction reports
artifacts/training-logs/                   all training + eval logs preserved
```

GGUF was skipped — `llama.cpp` isn't installed locally. `brew install llama.cpp`
plus a re-run of `civic-slm merge --skip-gguf=false` produces the .gguf.

## Recommended next steps

1. **Fix the synth extract prompt** to balance across all 5 schemas the
   eval bench uses. Re-synth a wider extraction set (need more Anthropic
   credit or a re-run with the local Gemma teacher).
2. **Crawl more aggressively** — `civic-slm crawl san-clemente --max 200
--since 2020-01-01` plus the other ~10 jurisdictions referenced in the
   bench. 34 chunks → 500+ chunks unlocks real CPT and rank/layer scaling.
3. **Re-run training cycle** with the larger corpus + balanced synth.
   Should genuinely beat base on at least factuality and extraction.
4. **Make `civic-slm doctor` warn on contamination.** Currently it surfaces
   only at eval time; surfacing during training prep would catch the same
   bug earlier.
