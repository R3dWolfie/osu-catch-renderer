# Recovered Catch clear: published-source candidate

This candidate starts at `55b58b14b29bec9f3fbd05022f88bc35085bff80`, the
Catch source in signed engine bundle 0.1.38. It carries the complete life-dip
repair from `1110dd9` plus the strict-judgement and Metal-parity changes reviewed
at `f0a1b0b`. It changes only failure classification in the GL and Metal render
paths, preserving the published AV1 encoder and CLI.

HP reaching zero or replay input stopping does not prove the play ended there.
When every generated fruit and large droplet has a caught-or-missed judgement,
the simulation continues to the end. A deficit, including just one missing
judgement on a short map, still permits an early failure. The 200 ms end grace,
NoFail parsing and simulation desync guard remain in force.

## Verification

- Generated regression tests against published source: **11 failed, 34 passed**.
  With this candidate: **45 passed**. Tests intercept immediately after the real
  simulation, before any renderer, audio or encoder starts.
- Broader software suite: **183 passed, 3 deselected**. The three excluded cases
  can open hardware encoder sessions. The pre-existing stale import in
  `test/test_leadin.py` remains an explicit excluded collection blocker.
- The exact privately retained Relax+HD report completes **810/810 objects**
  through both candidate GL and Metal simulation paths, with no truncation.
  Before the original fix, GL stopped at **339/810** and failed its desync guard.
- Four critical baseline files (both render paths, AV1 helper and CLI) match
  their signed bundle members byte for byte. The production diff touches no
  AV1 options or encoding code.
- No video, hardware context or performance benchmark was run. The replay and
  beatmap are deliberately excluded from this commit. This does not add Metal
  Mirror support or establish raster fidelity.

Release still needs a reviewed bundle containing this complete patch and an
exact-input canary on an approved physical GPU. No bundle or service was changed.
