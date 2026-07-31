# AI Usage Log

The project rules require citing every use of AI tools (chatbots are allowed WITH
citation; autonomous coding agents are NOT allowed). Log entries as you go — do not
reconstruct later. Each member is responsible for their own rows.

| Date       | Person                 | Tool            | Task / What it was used for              | How the output was verified                             |
| ---------- | ---------------------- | --------------- | ---------------------------------------- | ------------------------------------------------------- |
| 2026-06-30 | (example)              | ChatGPT         | Drafted the sampling function skeleton   | Ran on sample data, checked row counts + ratios by hand |
| 2026-06-30 | Irai Kumaran Sivanesan | Claude - Sonnet | Formating readme.md and requirements.txt | Mannually check each line changed as per required       |
| 2026-06-30 | Nalluraj Babu          | Claude - Sonnet | Readme.md and requirements.txt.          | Mannually check each line changed as per required       |
| 2026-06-30 | Nalluraj Babu          | Claude - Sonnet | Researched about existing unsupervised models | Understood how it works and learns pattern from the unlabelled data|
| 2026-07-13 | Nalluraj Babu          | Claude - Opus   | Understanding the Phase 2 spec, clarified the two-stage IDS idea, why anomaly detection is unsupervised, and how packet vs flow data differ | Re-read the project PDF against the explanation; confirmed my understanding matched Task 2.1/2.2 before coding |
| 2026-07-13 | Irai Kumaran Sivanesan | Claude - Sonnet | Learned how to keep the CIC session cookie apart from the code | Confirmed the token is gone from tracked source, `.env` is git-ignored, and the scripts still load the cookie correctly |
| 2026-07-14 | Nalluraj Babu          | Claude - Opus | I built the Phase 2 pipeline; used Claude to review and sanity-check my implementation | Verified against my run - confirmed array shapes, split sizes, and metric outputs on the seed-42 data matched what I expected |
| 2026-07-22 | Irai Kumaran Sivanesan | Claude - Opus  | Learned how to make ensemble union also the recheck of phase 2 to confirm the efficency coded | Made sure to check code and the method used to for finding and reverifying the result |
| 2026-07-29 | Nalluraj Babu          | Claude - Opus  | Writing assistance for the Phase 2 / Phase 4 presentation slides; all numbers are our own results | Checked every figure on the slides against phase2_metrics.json / phase3_recheck.json |

