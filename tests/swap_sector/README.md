# Sector swap test case

Experimental, test-only implementation. No database writes, no PCI changes, no production app integration.
Only `fetch_data.py` touches the database (read-only).

## Model (active: `method="pattern"`)

**Hypothesis:** the cells (PCIs) of one carrier are connected to the wrong antenna endpoints.
An endpoint = configured azimuth + antenna pattern (+ tilt) and moves as a whole. PCI numbers never change.

1. **Group** = site + operator + technology + EARFCN. Each operator + technology is run on its own.
2. **Expected profile — no drive-test RF values** (`antenna_profile.py`). Carrier frequency from the drive-test
   EARFCN (3GPP TS 36.101 band table). Pattern source, strict order, carried on every result:
   | Quality | Source | Highest verdict |
   |---|---|---|
   | `EXACT` | real antenna model (+ port) in the site table and an in-band `.pap` file | Confirmed (rule-based) |
   | `ASSUMED` | technology rule (LTE → CCVVPX308, NR → K800109221) with an in-band `.pap` file | Probable |
   | `APPROXIMATE` | generic 3GPP sector pattern (LTE TR 36.814, NR TR 38.901) | Probable |

   A vendor file for another frequency range (e.g. the 1710–1880 MHz file for B40) is **never** used.
   Missing azimuth, tilt or frequency → `NOT_TESTABLE`. The pattern is rotated to the configured azimuth and
   averaged into 36 × 10° bins. The file for the electrical tilt already contains that tilt (nearest tilt is
   reported); mechanical tilt and per-location elevation are not modelled yet.
3. **Observed side — same-location comparisons** (`pattern_detector.py`): at each drive-test location, which
   sector was stronger (two sectors ≥ 3 dB apart, or the serving sector against group sectors it did not report).
   Comparing at the same spot cancels distance, propagation and the common power level.
4. **Score** every assignment of endpoints to PCIs: mean dB by which the expected gains in that location's bin put
   the weaker sector ahead (beyond 3 dB). Report configured vs best mapping, improvement, bootstrap support.
5. **Decide:** `NORMAL` · `PROBABLE_SWAP` (≥ 2 sectors exchanged, clearly and stably better, every moved sector
   improved, same answer on alternating bins, no handover contradiction) · `CONFIRMED_SWAP` (also EXACT patterns,
   eNodeB-ID match and handover support) · `AZIMUTH_MISMATCH` (direction / RF anomaly: nothing fits, or one sector
   off with no exchange) · `AMBIGUOUS` · `NOT_ENOUGH_DATA` · `NOT_TESTABLE`.

Why not compare each PCI's profile shape on its own: in real drive tests a PCI is mostly measured only where it
serves, and removing its own power offset there makes a wrong antenna look as good as the right one. On project 193
that design found 3/66 injected swaps; it is kept only as a displayed diagnostic.

## Results on the local project-193 snapshot (fetched 11 Sep 2026, 23 sessions)

- **Carrier frequency:** EARFCN → band agrees with the phone's band label for all 561 linked sectors.
- **Antenna inputs:** the site table has tilts but **no antenna model or port**, so no carrier can be `EXACT`
  and no swap can be Confirmed. B3 carriers use the ASSUMED CCVVPX308 file; B1/B5/B8/B40/B41 use generic 3GPP.
- **Real configuration (all operators):** Normal 103 · Probable swap 7 · Direction / RF anomaly 23 ·
  Ambiguous 10 · Insufficient data 15 · Not testable 146 (single sector on the carrier).
- **Probable swaps (field-check candidates):** Vi 420230 B3, JIO 7492 B40, JIO 2139 B40, Vi 461003 B8,
  Vi 430572 B8, JIO 13383 B40, JIO 15753 B40. For 13383 and 15753 the few handover spots (4 each, below the
  5-spot minimum) point against the exchange.
- **Synthetic test** (whole endpoints exchanged in the config, real measurements, ±15° azimuth jitter):
  found 40/66, wrong sectors 1, false alarms 4/67 (3 of them on carriers the real config already flags).
  Held-out sites: 4/8 found, 0/12 false alarms. Other settings (max violation 2–4 dB, tolerance 1.5–6 dB,
  support 0.8): 38–43 found, 3–4 false alarms. Earlier methods: legacy bins 13/66, `evidence.py` combined 7/66.
- **Demo** (controlled synthetic, RF generated from the selected patterns): 5/5 correct.

## Limits

- No field-labelled swaps; one project. Thresholds are starting values from RF reasoning, not calibrated.
- Synthetic swaps test recovery of crossed configurations, not every physical feeder-swap effect.
- NR is not testable: drive-test NR rows carry placeholder cell ids.
- Per-location elevation and mechanical tilt are not in the expected profile.
- Handover points are serving-cell transition proxies, not protocol-confirmed handovers.

## Run (PowerShell, from `ML`)

```powershell
.\venv\Scripts\python.exe -m tests.swap_sector.fetch_data --project-id 193 --region india   # read-only DB
.\venv\Scripts\python.exe -m tests.swap_sector.build_dataset
.\venv\Scripts\python.exe -m tests.swap_sector.make_synthetic_swap
.\venv\Scripts\python.exe -m tests.swap_sector.make_demo
.\venv\Scripts\python.exe -m tests.swap_sector.detect_sector_swap --config real --operator Airtel --technology LTE
.\venv\Scripts\python.exe -m tests.swap_sector.detect_sector_swap --config demo
.\venv\Scripts\python.exe -m tests.swap_sector.validate
.\venv\Scripts\python.exe -m unittest tests.swap_sector.test_pattern_detector tests.swap_sector.test_evidence
.\venv\Scripts\python.exe -m streamlit run tests\swap_sector\swap_sector_dashboard.py --server.port 8501
```

`make_demo --source real` builds the demo from real project carriers instead (fails explicitly when too few qualify).
