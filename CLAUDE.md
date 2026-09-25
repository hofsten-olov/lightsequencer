# Light Sequencer DMX

PyQt5 desktop app for beat-synced DMX light sequencing. Main file: `lightSequencerDMX_full_v2.8.py`. Beat detection: `beat_sync.py`.

## Stack

- Python 3.11, PyQt5, NumPy, sounddevice, pyftdi
- DMX output via FT232R USB bit-bang (libusb-win32 driver, URL `ftdi://::/1`)
- Config persisted as JSON (`sequencer_config.json`)
- Fixture definitions in `units.txt`

## Architecture

- `Sequencer` — clock thread emitting step signals at BPM rate (hybrid sleep/spin-wait timing)
- `DmxSender` — background thread streaming DMX512 frames via FTDI serial
- `BeatDetector` (in `beat_sync.py`) — mic input via sounddevice, spectral-flux onset detection + autocorrelation BPM estimation
- `MainWindow` — PyQt5 UI wiring everything together; grid is `2*len(CHANNELS)` rows × 8 columns (two halves, one row per color channel); `CHANNELS` and `CHANNEL_COLORS` at the top of the file are the single source of truth for color rows — add a new light color there and every panel (grid, gate sliders, paint buttons, channel config) picks it up automatically
- Per-color DMX addressing: each color channel has its own DMX base address and unit type
- Per-color envelope: each channel has a ramp-up/duration/ramp-down triple (`ChannelGateSliders`, percentages of the step interval) instead of a single on/off gate; `MainWindow._trigger_envelope`/`_update_envelopes` run a shared timer that fades DMX output (via `_dmx_push_levels`) rather than snapping channels on/off — see `_normalize_gate_triples` for the legacy single-percent format this replaced

## Key conventions

- Versioned files (`v2.3`, `v2.4`, etc.) exist for history; the latest (`v2.8`) is the working copy
- BPM is never stored in pattern slots (only beat sync governs tempo)
- Pattern slots store full scenes: pattern grid, gates, DMX config, per-color channel config
- The sequencer uses eighth-note step pacing: `step_interval = 60 / bpm / 2`

## Agents

### beat-sync-researcher
Use this agent when debugging or improving `beat_sync.py` — onset detection, BPM estimation, phase alignment, latency compensation, or audio input issues. The agent should research the current implementation and propose targeted fixes.

### dmx-debugger
Use this agent when debugging DMX output issues — channel mapping, address offsets, fixture control, frame timing, or FTDI communication. The agent should trace the DMX data flow from pattern/probe through to `DmxSender`.

### ui-layout-inspector
Use this agent when investigating or modifying PyQt5 UI layout, widget wiring, signal/slot connections, or grid display logic. The agent should read the relevant widget classes and `MainWindow` setup.
