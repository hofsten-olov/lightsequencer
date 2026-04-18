# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 11:54:52 2026

@author: OlovvonHofsten
"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Light Sequencer – DMX + Beat Sync (v2.7)
- 16 steps (two full paces)
- Slots store/load FULL SCENES (pattern, gates, DMX config, per-color DMX personalities)
  (BPM/beat-phase NOT in slot; beat sync governs that)
- Unified config: 'sequencer_config.json' auto-loads on startup if present (no auto-save)
- Beat Sync panel under Pattern selector; latency slider -500..+500 ms with LIVE re-phase (BPM unchanged)

v2.7 New Features:
- Manual paint buttons (R/G/B/W): press-and-hold lights the color AND paints the
  current step into the pattern. Release stops painting; cells remain in the pattern.
- Clear All button to wipe the pattern
- Configurable step count (4..16)
- Continuous Auto Sync toggle (beat-sync tracks BPM and phase continuously)
- Fix: Stop button now also turns off DMX channels

v2.6 Features:
- Per-color DMX start address configuration (no longer hardcoded to 1)
- Unit type dropdown selector per color channel
- Dynamic channel function names loaded from units.txt
- units.txt file format for configurable fixture definitions

v2.5 Grid Layout Change:
- Grid is now 8x8 instead of 16x4
- Top 4 rows show steps 1-8 (Red, Green, Blue, White)
- Bottom 4 rows show steps 9-16 (Red, Green, Blue, White)
- Pattern playback: steps 1-8 on top half, then jumps to bottom half for steps 9-16

v2.4 Latency Fixes:
- Beat detection polling reduced from 100ms to 33ms for tighter sync
- Sequencer timing uses hybrid sleep/spin-wait for sub-millisecond precision
- DMX break timing uses spin-wait for accurate microsecond delays
"""

import sys, time, threading, json, os, math, atexit
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

from beat_sync import BeatDetector

CONFIG_FILE = "sequencer_config.json"
UNITS_FILE = "units.txt"
CHANNELS = ["Red", "Green", "Blue", "White"]
SLOTS = list(range(1, 10))

# ---------------- Unit Definition Loader ----------------

def load_units_from_file(filepath: str) -> dict:
    """
    Parse units.txt file and return dict of unit definitions.
    Format:
        [Unit Name]
        channel_number = Function Name
        ...
    Returns: {unit_name: {channel_num: function_name, ...}, ...}
    """
    units = {}
    if not os.path.exists(filepath):
        return units

    current_unit = None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Check for unit header [Unit Name]
                if line.startswith("[") and line.endswith("]"):
                    current_unit = line[1:-1].strip()
                    units[current_unit] = {}
                elif current_unit and "=" in line:
                    # Parse channel = function
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        try:
                            ch_num = int(parts[0].strip())
                            func_name = parts[1].strip()
                            units[current_unit][ch_num] = func_name
                        except ValueError:
                            pass  # Skip invalid lines
    except Exception as e:
        print(f"Warning: Could not load units file: {e}")

    return units


# Global units dictionary - loaded at startup
UNIT_DEFINITIONS = {}

def get_unit_names() -> list:
    """Return list of available unit names."""
    return list(UNIT_DEFINITIONS.keys())

def get_unit_channels(unit_name: str) -> dict:
    """Return channel mapping for a unit: {channel_num: function_name}"""
    return UNIT_DEFINITIONS.get(unit_name, {})

def get_function_to_channel(unit_name: str) -> dict:
    """Return {function_name_key: channel_num} for a unit."""
    channels = get_unit_channels(unit_name)
    # Convert function names to keys (lowercase, replace spaces/dashes with underscores)
    result = {}
    for ch, name in channels.items():
        key = name.lower().replace(" ", "_").replace("-", "_").replace("‑", "_")
        result[key] = ch
    return result

def get_channel_to_function(unit_name: str) -> dict:
    """Return {channel_num: function_name} for a unit."""
    return get_unit_channels(unit_name)


# ---------------- Precision Timing Helpers ----------------

# Global shutdown flag for spin-wait loops
_shutdown_flag = threading.Event()

def _spin_wait_until(target_time: float, stop_event: threading.Event = None):
    """Spin-wait until target_time (perf_counter). Use for sub-ms precision.
    Checks stop_event and global _shutdown_flag to allow clean exit."""
    while time.perf_counter() < target_time:
        if _shutdown_flag.is_set():
            return
        if stop_event is not None and stop_event.is_set():
            return

def _usleep(microseconds: float, stop_event: threading.Event = None):
    """Spin-wait for specified microseconds. More accurate than time.sleep() for short durations.
    Checks stop_event and global _shutdown_flag to allow clean exit."""
    end = time.perf_counter() + microseconds / 1_000_000.0
    while time.perf_counter() < end:
        if _shutdown_flag.is_set():
            return
        if stop_event is not None and stop_event.is_set():
            return

# ---------------- DMX Sender (streaming) ----------------

DMX_AVAILABLE = True
try:
    from pyftdi.serialext import serial_for_url
except Exception:
    DMX_AVAILABLE = False

class DmxSender(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    def __init__(self, ftdi_url='ftdi://::/1', fps=40):
        super().__init__()
        self.ftdi_url = ftdi_url
        self.fps = fps
        self.enabled = False
        self.frame = bytearray(512)
        self._lock = threading.Lock()
        self._ser = None
        self._thr = None
        self._stop = threading.Event()

    def set_url(self, url):
        self.ftdi_url = url

    def set_enabled(self, on: bool):
        if on == self.enabled:
            return
        self.enabled = on
        if on:
            self._start_stream()
        else:
            self._stop_stream()

    def update_channels(self, pairs):
        with self._lock:
            for ch, val in pairs:
                if 1 <= ch <= 512:
                    self.frame[ch-1] = max(0, min(int(val), 255))

    def _start_stream(self):
        self._stop.clear()
        if not DMX_AVAILABLE:
            self.status.emit("pyftdi not installed; DMX disabled.")
            return
        try:
            self._ser = serial_for_url(self.ftdi_url, baudrate=250000, bytesize=8,
                                       parity='N', stopbits=2, timeout=0)
        except Exception as e:
            self._ser = None
            self.status.emit(f"DMX open failed: {e}")
            return
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        self.status.emit(f"DMX streaming @ {self.fps} FPS via {self.ftdi_url}")

    def _stop_stream(self):
        self._stop.set()
        if self._thr:
            try:
                self._thr.join(timeout=1.0)
            except Exception:
                pass
        self._thr = None
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self.status.emit("DMX stopped")

    def stop(self):
        self.set_enabled(False)

    def _loop(self):
        period = 1.0 / float(self.fps)
        while not self._stop.is_set() and not _shutdown_flag.is_set():
            t0 = time.perf_counter()
            try:
                # DMX Break: 88-176µs (we use 120µs)
                self._ser.break_condition = True
                _usleep(120, self._stop)  # FIX #3: Use spin-wait instead of time.sleep()

                # Check for shutdown after usleep
                if self._stop.is_set() or _shutdown_flag.is_set():
                    break

                # DMX Mark After Break (MAB): 8-16µs (we use 12µs)
                self._ser.break_condition = False
                _usleep(12, self._stop)   # FIX #3: Use spin-wait instead of time.sleep()

                # Check for shutdown after usleep
                if self._stop.is_set() or _shutdown_flag.is_set():
                    break

                with self._lock:
                    payload = bytes([0x00]) + bytes(self.frame)
                self._ser.write(payload)
                self._ser.flush()
            except Exception as e:
                self.status.emit(f"DMX send error: {e}")
                self._stop.set()
                break

            # Maintain frame rate with hybrid sleep/spin approach
            dt = time.perf_counter() - t0
            remaining = period - dt
            if remaining > 0.002:
                # Use event wait instead of sleep for faster shutdown response
                self._stop.wait(timeout=remaining - 0.002)
            # Spin-wait for final precision (with stop check)
            target = t0 + period
            _spin_wait_until(target, self._stop)

# ---------------- Sequencer (16 steps) ----------------


class Sequencer(QtCore.QObject):
    step_changed  = QtCore.pyqtSignal(int, list)
    tempo_changed = QtCore.pyqtSignal(float)

    def __init__(self, beats=16, channels=CHANNELS, bpm=120.0):
        super().__init__()
        self.beats = beats
        self.channels = channels
        self.bpm = float(bpm)
        self.pattern = np.zeros((len(channels), beats), dtype=np.uint8)
        self.pattern[:, ::2] = 1
        self.playing = False
        self.step = -1

        # timing: store next tick time so we can nudge it without restarting
        self._next_tick_at = None
        self._stop_evt = threading.Event()
        self._thread = None

    def start(self):
        if self.playing and self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self.playing = True
        # set next tick right now; first loop will advance to step 0 immediately
        self._next_tick_at = time.perf_counter()
        self._thread = threading.Thread(target=self._clock_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.playing = False
        self._stop_evt.set()
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        self._thread = None

    def set_beats(self, beats):
        """Change step count, preserving existing pattern data."""
        old = self.beats
        self.beats = beats
        new_pattern = np.zeros((len(self.channels), beats), dtype=np.uint8)
        copy_cols = min(old, beats)
        new_pattern[:, :copy_cols] = self.pattern[:, :copy_cols]
        self.pattern = new_pattern
        if self.step >= beats:
            self.step = beats - 1

    def set_bpm(self, bpm):
        self.bpm = float(bpm)
        # when tempo changes, re-schedule next tick to keep continuity (no restart)
        if self._next_tick_at is not None:
            now = time.perf_counter()
            # pull next tick toward now by at most half a step to avoid long pauses
            step = self.step_interval_seconds(self.bpm)
            self._next_tick_at = min(self._next_tick_at, now + step)
        self.tempo_changed.emit(self.bpm)

    @staticmethod
    def step_interval_seconds(bpm):
        # keep eighth-note pacing
        return 60.0 / bpm / 2.0

    def nudge_phase_ms(self, delta_ms: float):
        """
        Advance (negative delta) or delay (positive delta) the next tick without changing the current step index.
        Does not restart the bar.
        """
        if self._next_tick_at is None:
            return
        self._next_tick_at += (delta_ms / 1000.0)

    def _clock_loop(self):
        # ensure we emit initial step 0 immediately, then schedule next ticks
        if self.step != -1:
            # if we were already mid-run and got re-started, do not double-emit
            pass
        else:
            # Emit step 0
            self.step = (self.step + 1) % self.beats
            self._emit_step()

            # And schedule the next tick cleanly in the future
            self._next_tick_at = time.perf_counter() + self.step_interval_seconds(self.bpm)

        while not self._stop_evt.is_set() and not _shutdown_flag.is_set():
            if not self.playing:
                # Use wait with timeout instead of sleep for faster shutdown response
                self._stop_evt.wait(timeout=0.01)
                continue

            now = time.perf_counter()
            if self._next_tick_at is None:
                self._next_tick_at = now + self.step_interval_seconds(self.bpm)

            time_until_tick = self._next_tick_at - now

            if time_until_tick <= 0:
                # Time to emit
                self.step = (self.step + 1) % self.beats
                self._emit_step()
                # schedule next tick
                self._next_tick_at += self.step_interval_seconds(self.bpm)
            elif time_until_tick > 0.003:
                # FIX #2: Hybrid sleep/spin-wait for precision
                # Sleep for bulk of time (leave 2ms for spin-wait)
                # Use wait with timeout for faster shutdown response
                self._stop_evt.wait(timeout=time_until_tick - 0.002)
            else:
                # Spin-wait for final 2-3ms for sub-millisecond precision
                _spin_wait_until(self._next_tick_at, self._stop_evt)

    def _emit_step(self):
        # Don't emit if shutting down
        if _shutdown_flag.is_set():
            return
        vals = [int(self.pattern[r, self.step]) for r in range(len(self.channels))]
        self.step_changed.emit(self.step, vals)

# ---------------- UI Widgets ----------------

class Grid(QtWidgets.QTableWidget):
    """
    8x8 Grid Layout for 16-step sequencer with 4 channels.

    Layout:
    - 8 rows, 8 columns
    - Top 4 rows (0-3): Steps 1-8 for Red, Green, Blue, White
    - Bottom 4 rows (4-7): Steps 9-16 for Red, Green, Blue, White

    Internal pattern storage remains (4 channels × 16 steps).
    Grid maps: row 0-3 col 0-7 → pattern[row, col]
               row 4-7 col 0-7 → pattern[row-4, col+8]
    """

    GRID_ROWS = 8  # 4 channels × 2 halves
    GRID_COLS = 8  # 8 steps per half

    def __init__(self, seq: Sequencer):
        super().__init__(self.GRID_ROWS, self.GRID_COLS)
        self.seq = seq
        self.num_channels = len(seq.channels)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self._last_highlighted_step = -1
        self._build()
        self.cellClicked.connect(self._toggle)

    def _grid_to_pattern(self, grid_row, grid_col):
        """Convert grid coordinates (row, col) to pattern coordinates (channel, step)."""
        if grid_row < self.num_channels:
            # Top half: rows 0-3 → channels 0-3, steps 0-7
            return (grid_row, grid_col)
        else:
            # Bottom half: rows 4-7 → channels 0-3, steps 8-15
            return (grid_row - self.num_channels, grid_col + 8)

    def _step_to_grid(self, step):
        """Convert sequencer step (0-15) to grid column and row offset."""
        if step < 8:
            # Steps 0-7: top half, column = step
            return (0, step)  # (row_offset, col)
        else:
            # Steps 8-15: bottom half, column = step - 8
            return (self.num_channels, step - 8)  # (row_offset, col)

    def _build(self):
        for r in range(self.GRID_ROWS):
            self.setRowHeight(r, 34)
            for c in range(self.GRID_COLS):
                self.setColumnWidth(c, 38)  # Wider cells since we have only 8 columns
                it = QtWidgets.QTableWidgetItem()
                it.setFlags(QtCore.Qt.ItemIsEnabled)
                self.setItem(r, c, it)
        self.update_step_count()

    def _is_valid_step(self, grid_row, grid_col):
        """Check if grid cell maps to a valid step within current beat count."""
        _, step = self._grid_to_pattern(grid_row, grid_col)
        return step < self.seq.beats

    def update_step_count(self):
        """Update grid visibility based on current beat count."""
        beats = self.seq.beats
        # Show/hide bottom half rows
        for r in range(self.num_channels, self.GRID_ROWS):
            self.setRowHidden(r, beats <= 8)
        self.refresh_all()

    def refresh_all(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                self._paint(r, c, False)

    def _get_row_color_index(self, grid_row):
        """Get the color index (0-3) for a grid row."""
        return grid_row % self.num_channels

    def _paint(self, grid_row, grid_col, hl):
        # Guard against missing items (can happen during init or cleanup)
        item = self.item(grid_row, grid_col)
        if item is None:
            return

        # Convert grid coords to pattern coords
        channel, step = self._grid_to_pattern(grid_row, grid_col)

        # Out-of-bounds steps: paint as disabled
        if step >= self.seq.beats:
            item.setBackground(QtGui.QBrush(QtGui.QColor(20, 20, 20)))
            item.setText("")
            return

        on = bool(self.seq.pattern[channel, step])

        # Base colors for active cells per channel (color repeats for top/bottom)
        color_idx = self._get_row_color_index(grid_row)
        row_colors = {
            0: QtGui.QColor(235, 60, 60),    # Red
            1: QtGui.QColor(60, 200, 90),    # Green
            2: QtGui.QColor(70, 100, 235),   # Blue
            3: QtGui.QColor(235, 235, 235)   # White
        }

        if on:
            base = row_colors.get(color_idx, QtGui.QColor(200, 200, 200))
        else:
            base = QtGui.QColor(35, 35, 35)

        if hl:
            # Highlight tint overlay
            highlight = QtGui.QColor(255, 255, 140)   # soft yellow
            color = QtGui.QColor(
                int(base.red()   * 0.6 + highlight.red()   * 0.4),
                int(base.green() * 0.6 + highlight.green() * 0.4),
                int(base.blue()  * 0.6 + highlight.blue()  * 0.4),
            )
            item.setBackground(QtGui.QBrush(color))
            item.setForeground(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        else:
            item.setBackground(QtGui.QBrush(base))
            item.setForeground(QtGui.QBrush(QtGui.QColor(20, 20, 20)))

        item.setText("")

    def _toggle(self, grid_row, grid_col):
        channel, step = self._grid_to_pattern(grid_row, grid_col)
        if step >= self.seq.beats:
            return
        self.seq.pattern[channel, step] = 0 if self.seq.pattern[channel, step] else 1
        self._paint(grid_row, grid_col, False)

    def set_step_highlight(self, step):
        """Highlight the current step across all channels."""
        # Clear previous highlight
        if self._last_highlighted_step >= 0 and self._last_highlighted_step != step:
            prev_row_offset, prev_col = self._step_to_grid(self._last_highlighted_step)
            for ch in range(self.num_channels):
                self._paint(prev_row_offset + ch, prev_col, False)

        # Set new highlight
        row_offset, col = self._step_to_grid(step)
        for ch in range(self.num_channels):
            self._paint(row_offset + ch, col, True)

        self._last_highlighted_step = step


class PreviewStrip(QtWidgets.QWidget):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.values = [0] * len(channels)
        self.setMinimumHeight(40)

    def set_values(self, values):
        self.values = values[:]
        self.update()

    def clear_values(self):
        self.values = [0] * len(self.channels)
        self.update()

    def clear_channel(self, idx):
        if 0 <= idx < len(self.values):
            self.values[idx] = 0
            self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        w = self.width()
        h = self.height()
        n = len(self.channels)
        bw = max(10, int(w / n))
        cols = [
            QtGui.QColor(235, 60, 60),
            QtGui.QColor(60, 200, 90),
            QtGui.QColor(70, 100, 235),
            QtGui.QColor(235, 235, 235)
        ]
        for i, _ in enumerate(self.channels):
            col = cols[i] if self.values[i] else QtGui.QColor(30, 30, 30)
            p.fillRect(i * bw, 0, bw - 2, h, col)


class ChannelGateSliders(QtWidgets.QWidget):
    value_changed = QtCore.pyqtSignal(int, float)

    def __init__(self, channels, initial=50):
        super().__init__()
        self.sliders = []
        self.labels = []
        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel("Gate per row (%)"), 0, 0, 1, 3)
        row_cols = {
            "Red": "rgb(235,60,60)",
            "Green": "rgb(60,200,90)",
            "Blue": "rgb(70,100,235)",
            "White": "rgb(235,235,235)"
        }
        for i, ch in enumerate(channels):
            lbl = QtWidgets.QLabel(ch)
            lbl.setStyleSheet(f"color:{row_cols.get(ch, 'rgb(200,200,200)')};")
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(1, 100)
            s.setValue(initial)
            v = QtWidgets.QLabel(f"{initial}%")
            v.setMinimumWidth(40)
            v.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.sliders.append(s)
            self.labels.append(v)
            r = i + 1
            lay.addWidget(lbl, r, 0)
            lay.addWidget(s, r, 1)
            lay.addWidget(v, r, 2)
            s.valueChanged.connect(
                lambda val, idx=i: (self.labels[idx].setText(f"{val}%"), self.value_changed.emit(idx, float(val)))
            )

    def gates(self):
        return [float(s.value()) for s in self.sliders]

    def set_gates(self, values):
        for i, v in enumerate(values):
            self.sliders[i].setValue(int(v))
            self.labels[i].setText(f"{int(v)}%")


class PatternSlots(QtWidgets.QWidget):
    slot_changed = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.btn_group = QtWidgets.QButtonGroup(self)
        self.btn_group.setExclusive(True)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel("Pattern:"))
        for n in SLOTS:
            b = QtWidgets.QPushButton(str(n))
            b.setCheckable(True)
            b.setFixedWidth(28)
            self.btn_group.addButton(b, n)
            lay.addWidget(b)
        self.btn_group.button(1).setChecked(True)
        self.btn_group.buttonClicked[int].connect(self._on_slot_clicked)
        self.save_btn = QtWidgets.QPushButton("Save")
        self.load_btn = QtWidgets.QPushButton("Load")
        lay.addSpacing(8)
        lay.addWidget(self.save_btn)
        lay.addWidget(self.load_btn)

    def _on_slot_clicked(self, slot_id: int):
        self.slot_changed.emit(slot_id)


class DmxPanel(QtWidgets.QGroupBox):
    config_changed = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__("DMX Output")
        lay = QtWidgets.QGridLayout(self)
        self.enable_chk = QtWidgets.QCheckBox("Enable DMX Output")
        self.url_edit = QtWidgets.QLineEdit("ftdi://::/1")
        lay.addWidget(self.enable_chk, 0, 0, 1, 2)
        lay.addWidget(QtWidgets.QLabel("FTDI URL:"), 1, 0)
        lay.addWidget(self.url_edit, 1, 1)
        self.enable_chk.toggled.connect(self.config_changed.emit)
        self.url_edit.textChanged.connect(self.config_changed.emit)

    def config(self):
        return {"enabled": self.enable_chk.isChecked(), "url": self.url_edit.text().strip()}

    def set_config(self, cfg):
        self.enable_chk.setChecked(bool(cfg.get("enabled", False)))
        self.url_edit.setText(cfg.get("url", "ftdi://::/1"))


class ColorChannelConfig(QtWidgets.QGroupBox):
    """Panel for configuring DMX address and unit type per color channel."""
    config_changed = QtCore.pyqtSignal()

    def __init__(self, color_names: list, unit_names: list):
        super().__init__("Channel Configuration")
        self.color_names = color_names
        self.unit_names = unit_names

        lay = QtWidgets.QGridLayout(self)
        lay.setContentsMargins(4, 8, 4, 4)

        # Headers
        lay.addWidget(QtWidgets.QLabel("Color"), 0, 0)
        lay.addWidget(QtWidgets.QLabel("DMX Addr"), 0, 1)
        lay.addWidget(QtWidgets.QLabel("Unit Type"), 0, 2)

        self._addr_spinboxes = {}
        self._unit_combos = {}

        row_colors = {
            "Red": "rgb(235,60,60)",
            "Green": "rgb(60,200,90)",
            "Blue": "rgb(70,100,235)",
            "White": "rgb(235,235,235)"
        }

        for i, color in enumerate(color_names):
            # Color label
            lbl = QtWidgets.QLabel(color)
            lbl.setStyleSheet(f"color:{row_colors.get(color, 'rgb(200,200,200)')};")
            lay.addWidget(lbl, i + 1, 0)

            # DMX address spinbox (1-512)
            addr_spin = QtWidgets.QSpinBox()
            addr_spin.setRange(1, 512)
            addr_spin.setValue(1)
            addr_spin.setFixedWidth(60)
            self._addr_spinboxes[color] = addr_spin
            addr_spin.valueChanged.connect(self.config_changed.emit)
            lay.addWidget(addr_spin, i + 1, 1)

            # Unit type combo
            unit_combo = QtWidgets.QComboBox()
            unit_combo.addItems(unit_names if unit_names else ["(No units loaded)"])
            unit_combo.setMinimumWidth(150)
            self._unit_combos[color] = unit_combo
            unit_combo.currentTextChanged.connect(self.config_changed.emit)
            lay.addWidget(unit_combo, i + 1, 2)

    def get_config(self, color: str) -> dict:
        """Get config for a specific color: {dmx_addr, unit_type}"""
        return {
            "dmx_addr": self._addr_spinboxes.get(color, None).value() if color in self._addr_spinboxes else 1,
            "unit_type": self._unit_combos.get(color, None).currentText() if color in self._unit_combos else ""
        }

    def get_all_config(self) -> dict:
        """Get config for all colors: {color: {dmx_addr, unit_type}}"""
        return {color: self.get_config(color) for color in self.color_names}

    def set_config(self, color: str, dmx_addr: int, unit_type: str):
        """Set config for a specific color."""
        if color in self._addr_spinboxes:
            self._addr_spinboxes[color].setValue(int(dmx_addr))
        if color in self._unit_combos:
            idx = self._unit_combos[color].findText(unit_type)
            if idx >= 0:
                self._unit_combos[color].setCurrentIndex(idx)

    def set_all_config(self, config: dict):
        """Set config for all colors from dict."""
        for color, cfg in (config or {}).items():
            if isinstance(cfg, dict):
                self.set_config(
                    color,
                    cfg.get("dmx_addr", 1),
                    cfg.get("unit_type", "")
                )

    def update_unit_names(self, unit_names: list):
        """Update available unit names in all combos."""
        self.unit_names = unit_names
        for color, combo in self._unit_combos.items():
            current = combo.currentText()
            combo.clear()
            combo.addItems(unit_names if unit_names else ["(No units loaded)"])
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)


class DmxFunctionProbe(QtWidgets.QGroupBox):
    """Dynamic fixture controls panel that updates based on selected unit type."""
    test_start = QtCore.pyqtSignal(dict)
    test_stop = QtCore.pyqtSignal(dict)
    test_update = QtCore.pyqtSignal(dict)
    write_requested = QtCore.pyqtSignal(str, dict)
    read_requested = QtCore.pyqtSignal(str)

    def __init__(self, color_names=("Red", "Green", "Blue", "White"), default_unit: str = None):
        super().__init__("Fixture Controls")
        self.color_names = list(color_names)
        self._current_unit = default_unit or ""
        self._sliders = {}
        self._vals = {}
        self._channel_order = []  # List of (channel_num, function_name)

        self._main_layout = QtWidgets.QVBoxLayout(self)

        # Sliders area (will be rebuilt when unit changes)
        self._sliders_widget = QtWidgets.QWidget()
        self._sliders_layout = QtWidgets.QGridLayout(self._sliders_widget)
        self._main_layout.addWidget(self._sliders_widget)

        # Color selector and buttons
        btn_widget = QtWidgets.QWidget()
        btn_layout = QtWidgets.QVBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(QtWidgets.QLabel("Color row:"))
        self.color_select = QtWidgets.QComboBox()
        self.color_select.addItems(self.color_names)
        color_row.addWidget(self.color_select)
        color_row.addStretch()
        btn_layout.addLayout(color_row)

        test_row = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton("Start Test")
        self.btn_stop = QtWidgets.QPushButton("Stop Test")
        test_row.addWidget(self.btn_start)
        test_row.addWidget(self.btn_stop)
        btn_layout.addLayout(test_row)

        rw_row = QtWidgets.QHBoxLayout()
        self.btn_write = QtWidgets.QPushButton("Write to Color")
        self.btn_read = QtWidgets.QPushButton("Read from Color")
        rw_row.addWidget(self.btn_write)
        rw_row.addWidget(self.btn_read)
        btn_layout.addLayout(rw_row)

        self._main_layout.addWidget(btn_widget)

        # Connect buttons
        self.btn_start.clicked.connect(lambda: self.test_start.emit(self.values_by_channel()))
        self.btn_stop.clicked.connect(lambda: self.test_stop.emit(self.values_by_channel()))
        self.btn_write.clicked.connect(lambda: self.write_requested.emit(self.color_select.currentText(), self.values_by_channel()))
        self.btn_read.clicked.connect(lambda: self.read_requested.emit(self.color_select.currentText()))

        # Build initial sliders
        self._rebuild_sliders()

    def set_unit(self, unit_name: str):
        """Change the unit type and rebuild sliders."""
        if unit_name == self._current_unit:
            return
        self._current_unit = unit_name
        self._rebuild_sliders()

    def _rebuild_sliders(self):
        """Rebuild slider widgets based on current unit."""
        # Clear existing sliders
        self._sliders.clear()
        self._vals.clear()
        self._channel_order.clear()

        # Clear layout
        while self._sliders_layout.count():
            item = self._sliders_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get channel definitions for current unit
        channels = get_unit_channels(self._current_unit)

        if not channels:
            # No unit selected or empty - show placeholder
            lbl = QtWidgets.QLabel("Select a unit type to see controls")
            lbl.setStyleSheet("color: gray; font-style: italic;")
            self._sliders_layout.addWidget(lbl, 0, 0, 1, 3)
            self.setTitle("Fixture Controls")
            return

        self.setTitle(f"Fixture Controls - {self._current_unit}")

        # Headers
        self._sliders_layout.addWidget(QtWidgets.QLabel("Function"), 0, 0)
        self._sliders_layout.addWidget(QtWidgets.QLabel("Value"), 0, 1)
        self._sliders_layout.addWidget(QtWidgets.QLabel("Cur"), 0, 2)

        # Sort channels by number
        sorted_channels = sorted(channels.items(), key=lambda x: x[0])

        for row, (ch_num, func_name) in enumerate(sorted_channels, start=1):
            self._channel_order.append((ch_num, func_name))

            # Label
            lbl = QtWidgets.QLabel(f"{ch_num}: {func_name}")
            self._sliders_layout.addWidget(lbl, row, 0)

            # Slider
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(0, 255)
            s.setValue(0)
            s.setTickPosition(QtWidgets.QSlider.TicksBelow)
            s.setTickInterval(16)
            self._sliders[ch_num] = s
            self._sliders_layout.addWidget(s, row, 1)

            # Value label
            val_lbl = QtWidgets.QLabel("0")
            val_lbl.setMinimumWidth(36)
            self._vals[ch_num] = val_lbl
            self._sliders_layout.addWidget(val_lbl, row, 2)

            # Connect slider
            s.valueChanged.connect(
                lambda v, ch=ch_num: (self._vals[ch].setText(str(v)), self.test_update.emit(self.values_by_channel()))
            )

    def values_by_channel(self) -> dict:
        """Return {channel_num: value} for all sliders."""
        return {ch: s.value() for ch, s in self._sliders.items()}

    def set_from_channel_map(self, ch_map: dict):
        """Set slider values from {channel_num: value} dict."""
        for ch, s in self._sliders.items():
            val = ch_map.get(ch, 0) if ch_map else 0
            s.setValue(int(val))
            self._vals[ch].setText(str(int(val)))


class BeatSyncPanel(QtWidgets.QGroupBox):
    sync_requested = QtCore.pyqtSignal()
    latency_changed = QtCore.pyqtSignal(int)
    auto_sync_toggled = QtCore.pyqtSignal(bool)

    def __init__(self):
        super().__init__("Beat Sync")
        lay = QtWidgets.QGridLayout(self)
        self.bpm_lbl = QtWidgets.QLabel("BPM: 0.00")
        self.conf_lbl = QtWidgets.QLabel("Confidence: 0.00")
        self.lat_sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.lat_sld.setRange(-500, 500)
        self.lat_sld.setValue(0)
        self.lat_sld.setTickInterval(25)
        self.lat_val = QtWidgets.QLabel("0 ms")
        self.lat_val.setMinimumWidth(60)
        self.sync_btn = QtWidgets.QPushButton("Sync Now (phase align)")
        self.auto_chk = QtWidgets.QCheckBox("Auto Sync (continuous)")
        lay.addWidget(self.bpm_lbl, 0, 0, 1, 2)
        lay.addWidget(self.conf_lbl, 1, 0, 1, 2)
        lay.addWidget(QtWidgets.QLabel("Latency (ms):"), 2, 0)
        lay.addWidget(self.lat_sld, 2, 1)
        lay.addWidget(self.lat_val, 2, 2)
        lay.addWidget(self.sync_btn, 3, 0, 1, 3)
        lay.addWidget(self.auto_chk, 4, 0, 1, 3)
        self.lat_sld.valueChanged.connect(self._on_latency_changed)
        self.sync_btn.clicked.connect(self.sync_requested.emit)
        self.auto_chk.toggled.connect(self.auto_sync_toggled.emit)

    def is_auto_sync_on(self) -> bool:
        return self.auto_chk.isChecked()

    def _on_latency_changed(self, v: int):
        self.lat_val.setText(f"{int(v):+d} ms")
        self.latency_changed.emit(int(v))

    def set_bpm(self, bpm: float):
        self.bpm_lbl.setText(f"BPM: {bpm:.2f}")

    def set_conf(self, conf: float):
        self.conf_lbl.setText(f"Confidence: {conf:.2f}")

    def latency_ms(self) -> int:
        return int(self.lat_sld.value())


# ---------------- Kick Tuning Dialog (v2.7) ----------------

class KickTuningDialog(QtWidgets.QDialog):
    """Live-tuning dialog for the kick (low-band) onset detector and paint latency.
    Non-modal: stays open while you play and paint."""

    def __init__(self, detector, get_offset_ms, set_offset_ms, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.kick = detector.kick
        self._get_offset_ms = get_offset_ms
        self._set_offset_ms = set_offset_ms
        self._last_count = self.kick.onset_count

        self.setWindowTitle("Kick Detector Tuning")
        self.setMinimumWidth(420)

        lay = QtWidgets.QVBoxLayout(self)

        # Live readout
        readout = QtWidgets.QHBoxLayout()
        self.onset_lbl = QtWidgets.QLabel("Onsets: 0")
        self.onset_lbl.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.flash = QtWidgets.QLabel("  ")
        self.flash.setFixedSize(24, 24)
        self.flash.setStyleSheet("background-color: rgb(50,50,55); border-radius: 12px;")
        readout.addWidget(self.flash)
        readout.addWidget(self.onset_lbl)
        readout.addStretch()
        lay.addLayout(readout)

        # Sliders
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1, 1)
        self._sliders = {}
        params = [
            # key,           label,                      min,   max, default,                         divisor, unit
            ("hp",           "High-pass (reject below)",   0,  1000, int(self.kick.hp_hz),                 1, "Hz"),
            ("cutoff",       "Low-pass (reject above)",   50,  1500, int(self.kick.cutoff_hz),             1, "Hz"),
            ("threshold_k",  "Onset threshold (k)",        5,   100, int(self.kick.k * 10),               10, ""),
            ("debounce_ms",  "Debounce",                  20,   500, int(self.kick.debounce_s*1000),       1, "ms"),
            ("paint_offset", "Paint latency offset",       0,   500, int(self._get_offset_ms()),           1, "ms"),
        ]
        for row, (key, label, lo, hi, default, div, unit) in enumerate(params):
            grid.addWidget(QtWidgets.QLabel(label), row, 0)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(default)
            val_text = f"{default / div:.2f}" if div > 1 else str(default)
            val_lbl = QtWidgets.QLabel(f"{val_text} {unit}".strip())
            val_lbl.setMinimumWidth(80)
            val_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(s, row, 1)
            grid.addWidget(val_lbl, row, 2)
            self._sliders[key] = (s, val_lbl, div, unit)
            s.valueChanged.connect(lambda v, k=key: self._on_changed(k))

        lay.addLayout(grid)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset Detector")
        reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        # Live refresh timer
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _on_changed(self, key):
        s, val_lbl, div, unit = self._sliders[key]
        raw = s.value()
        val = raw / div
        val_text = f"{val:.2f}" if div > 1 else str(raw)
        val_lbl.setText(f"{val_text} {unit}".strip())

        if key == "cutoff":
            self.kick.set_cutoff(raw)
        elif key == "hp":
            self.kick.set_hp(raw)
        elif key == "threshold_k":
            self.kick.k = val
        elif key == "debounce_ms":
            self.kick.debounce_s = raw / 1000.0
        elif key == "paint_offset":
            self._set_offset_ms(raw)

    def _refresh(self):
        count = self.kick.onset_count
        self.onset_lbl.setText(f"Onsets: {count}")
        if count != self._last_count:
            self._last_count = count
            # flash
            self.flash.setStyleSheet("background-color: rgb(255,220,50); border-radius: 12px;")
            QtCore.QTimer.singleShot(
                80, lambda: self.flash.setStyleSheet(
                    "background-color: rgb(50,50,55); border-radius: 12px;"
                )
            )

    def _on_reset(self):
        self.kick.reset()
        self._last_count = 0

    def showEvent(self, e):
        # Resync counter so we don't fire a stale flash on reopen
        self._last_count = self.kick.onset_count
        if not self._timer.isActive():
            self._timer.start()
        super().showEvent(e)

    def closeEvent(self, e):
        self._timer.stop()
        e.accept()


# ---------------- Main Window (v2.7) ----------------

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Light Sequencer – DMX + Beat Sync (v2.7)")
        self._closing = False
        self._probe_active = False
        self._last_probe_channels = []

        # State
        self.seq = Sequencer(beats=16, channels=CHANNELS, bpm=120)
        self.dmx = DmxSender()
        self.color_names = CHANNELS[:]
        self.active_colors = [0, 0, 0, 0]
        self.last_active_colors = [0, 0, 0, 0]
        self.color_dmx_map = {name: {} for name in self.color_names}
        self.slots = {str(n): None for n in SLOTS}
        self._last_latency_ms = 0

        # UI & layout
        self.grid = Grid(self.seq)
        self.preview = PreviewStrip(self.seq.channels)
        self.bpm_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bpm_slider.setRange(30, 300)
        self.bpm_slider.setValue(int(self.seq.bpm))
        self.bpm_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.bpm_slider.setTickInterval(10)
        self.bpm_slider.setSingleStep(1)
        self.bpm_label = QtWidgets.QLabel(f"{int(self.seq.bpm)} BPM")
        self.bpm_label.setMinimumWidth(80)
        self.play_btn = QtWidgets.QPushButton("Play")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(4, 16)
        self.steps_spin.setValue(16)
        self.steps_spin.setSuffix(" steps")

        self.gate_panel = ChannelGateSliders(self.seq.channels, initial=50)
        self.slots_panel = PatternSlots()
        self.beat_panel = BeatSyncPanel()
        self.dmx_panel = DmxPanel()

        # New: Channel config panel with unit selection
        unit_names = get_unit_names()
        self.channel_config = ColorChannelConfig(self.color_names, unit_names)

        # Probe panel - default to first unit if available
        default_unit = unit_names[0] if unit_names else ""
        self.probe_panel = DmxFunctionProbe(self.color_names, default_unit)

        self.btn_save_config = QtWidgets.QPushButton("Save Config File…")
        self.btn_load_config = QtWidgets.QPushButton("Load Config File…")
        self.btn_reload_units = QtWidgets.QPushButton("Reload Units")
        self._align_timer = None

        controls = QtWidgets.QGridLayout()
        controls.addWidget(QtWidgets.QLabel("Tempo:"), 0, 0)
        controls.addWidget(self.bpm_slider, 0, 1)
        controls.addWidget(self.bpm_label, 0, 2)
        controls.addWidget(self.play_btn, 0, 3)
        controls.addWidget(self.stop_btn, 0, 4)
        controls.addWidget(QtWidgets.QLabel("Steps:"), 0, 5)
        controls.addWidget(self.steps_spin, 0, 6)
        self.clear_btn = QtWidgets.QPushButton("Clear All")
        self.clear_btn.setToolTip("Clear all pattern cells in the grid")
        controls.addWidget(self.clear_btn, 0, 7)
        controls.setColumnStretch(1, 1)

        # --- Paint buttons (press-and-hold) + per-color "Paint to kick" toggles ---
        paint_grid = QtWidgets.QGridLayout()
        paint_grid.addWidget(QtWidgets.QLabel("Paint:"), 0, 0)
        self.paint_btns = []
        self.kick_chks = []
        paint_colors = {
            "Red":   ("rgb(235,60,60)",   "rgb(150,30,30)"),
            "Green": ("rgb(60,200,90)",   "rgb(30,120,50)"),
            "Blue":  ("rgb(70,100,235)",  "rgb(40,60,150)"),
            "White": ("rgb(235,235,235)", "rgb(140,140,140)"),
        }
        for idx, ch_name in enumerate(self.color_names):
            btn = QtWidgets.QPushButton(ch_name)
            btn.setMinimumHeight(40)
            btn.setAutoRepeat(False)
            up, down = paint_colors.get(ch_name, ("rgb(200,200,200)", "rgb(100,100,100)"))
            fg = "black" if ch_name == "White" else "white"
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {up}; color: {fg}; font-weight: bold; border-radius: 4px; }}"
                f"QPushButton:pressed {{ background-color: {down}; }}"
            )
            btn.setToolTip(f"Hold to light {ch_name} and paint the current step(s)")
            paint_grid.addWidget(btn, 0, idx + 1)
            self.paint_btns.append(btn)
            btn.pressed.connect(lambda i=idx: self._on_paint_pressed(i))
            btn.released.connect(lambda i=idx: self._on_paint_released(i))

            chk = QtWidgets.QCheckBox("Paint to kick")
            chk.setToolTip(f"Auto-paint this channel when a kick-drum transient is detected")
            paint_grid.addWidget(chk, 1, idx + 1, QtCore.Qt.AlignCenter)
            self.kick_chks.append(chk)

        # Make columns 1..4 share width evenly
        for c in range(1, 5):
            paint_grid.setColumnStretch(c, 1)

        # Paint state — one flag per channel
        self._paint_active = [False] * len(self.color_names)

        main = QtWidgets.QHBoxLayout()
        # Channel Configuration lives in a separate non-modal dialog (set-once panel).
        self.channel_config_dialog = QtWidgets.QDialog(self)
        self.channel_config_dialog.setWindowTitle("Channel Configuration")
        _cc_layout = QtWidgets.QVBoxLayout(self.channel_config_dialog)
        _cc_layout.addWidget(self.channel_config)
        self.btn_open_channel_config = QtWidgets.QPushButton("Channel Configuration…")
        self.btn_open_channel_config.clicked.connect(self._open_channel_config_dialog)

        self.btn_open_kick_tuning = QtWidgets.QPushButton("Kick Detector Tuning…")
        self.btn_open_kick_tuning.clicked.connect(self._open_kick_tuning_dialog)
        self._kick_tuning_dialog = None  # created lazily

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.gate_panel)
        left.addWidget(self.slots_panel)
        left.addWidget(self.beat_panel)
        left.addWidget(self.btn_open_channel_config)
        left.addWidget(self.btn_open_kick_tuning)
        left.addWidget(self.btn_save_config)
        left.addWidget(self.btn_load_config)
        left.addWidget(self.btn_reload_units)
        left.addStretch(1)

        # Right column — wider than natural so the Fixture Controls sliders
        # have comfortable room on lower-res screens.
        self.dmx_panel.setMinimumWidth(500)
        self.probe_panel.setMinimumWidth(500)
        right_container = QtWidgets.QWidget()
        right_container.setMinimumWidth(520)
        right = QtWidgets.QVBoxLayout(right_container)
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self.dmx_panel)
        right.addWidget(self.probe_panel)
        right.addStretch(1)

        main.addLayout(left, stretch=0)
        main.addWidget(self.grid, stretch=1)
        main.addWidget(right_container, stretch=0)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(controls)
        root.addLayout(paint_grid)
        root.addLayout(main)
        root.addWidget(self.preview)
        self.status_lbl = QtWidgets.QLabel()
        root.addWidget(self.status_lbl)

        # Timers & wiring
        self.gate_timers = [QtCore.QTimer(self) for _ in self.seq.channels]
        for i, t in enumerate(self.gate_timers):
            t.setSingleShot(True)
            t.timeout.connect(lambda idx=i: self._on_gate_timeout(idx))

        self.play_btn.clicked.connect(self._on_play_pressed)
        self.stop_btn.clicked.connect(self._on_stop_pressed)
        self.steps_spin.valueChanged.connect(self._on_steps_changed)
        self.clear_btn.clicked.connect(self._on_clear_all)
        self.bpm_slider.valueChanged.connect(lambda v: (self.seq.set_bpm(v), self._sync_bpm_label(v)))
        self.seq.tempo_changed.connect(lambda b: self._sync_bpm_label(b))
        self.seq.step_changed.connect(self.on_step)
        self.dmx_panel.config_changed.connect(self._apply_dmx_config)
        self.slots_panel.save_btn.clicked.connect(self.save_current_slot)
        self.slots_panel.load_btn.clicked.connect(self.load_current_slot)

        self.btn_save_config.clicked.connect(self.save_full_config)
        self.btn_load_config.clicked.connect(self.load_full_config)
        self.btn_reload_units.clicked.connect(self._reload_units)

        self.probe_panel.test_start.connect(self._on_probe_start)
        self.probe_panel.test_stop.connect(self._on_probe_stop)
        self.probe_panel.test_update.connect(self._on_probe_update)
        self.probe_panel.write_requested.connect(self._on_probe_write)
        self.probe_panel.read_requested.connect(self._on_probe_read)

        # Connect channel config changes
        self.channel_config.config_changed.connect(self._on_channel_config_changed)

        # Connect probe color selection to update probe panel unit
        self.probe_panel.color_select.currentTextChanged.connect(self._on_probe_color_changed)

        # Beat detector always on
        self.detector = BeatDetector()
        self.detector.start(device=None)

        # FIX #1: Reduced polling interval from 100ms to 33ms for tighter beat sync
        self.beat_timer = QtCore.QTimer(self)
        self.beat_timer.setInterval(33)  # ~30 Hz polling instead of 10 Hz
        self.beat_timer.timeout.connect(self._beat_refresh)
        self.beat_timer.start()

        self.beat_panel.sync_requested.connect(self._beat_sync_now)
        self.beat_panel.latency_changed.connect(self._on_latency_changed_live)
        self.beat_panel.auto_sync_toggled.connect(self._on_auto_sync_toggled)

        # Auto-sync timer (runs continuously when enabled)
        self._auto_sync_timer = QtCore.QTimer(self)
        self._auto_sync_timer.setInterval(500)  # check every 500ms
        self._auto_sync_timer.timeout.connect(self._auto_sync_tick)

        # Auto-paint (kick) polling: watches detector.kick.onset_count and
        # fires a paint pulse on enabled channels for each new onset.
        self._last_kick_count = self.detector.kick.onset_count
        self._kick_paint_offset_ms = 0  # mic/algorithm latency compensation
        self._kick_paint_timer = QtCore.QTimer(self)
        self._kick_paint_timer.setInterval(20)  # 50 Hz poll
        self._kick_paint_timer.timeout.connect(self._kick_paint_tick)
        self._kick_paint_timer.start()

        # Don't use aboutToQuit - it can cause double cleanup issues
        # Cleanup is handled in closeEvent and main() instead

        # Initialize & auto-load config if exists
        self.seq.step = -1
        # Don't emit step during init - can cause issues
        if os.path.exists(CONFIG_FILE):
            self._load_config_path(CONFIG_FILE)

    # ---------- Unit management ----------

    def _reload_units(self):
        """Reload unit definitions from units.txt."""
        global UNIT_DEFINITIONS
        UNIT_DEFINITIONS = load_units_from_file(UNITS_FILE)
        unit_names = get_unit_names()
        self.channel_config.update_unit_names(unit_names)
        self._set_status(f"Reloaded {len(unit_names)} units from {UNITS_FILE}")

        # Update probe panel if a color is selected
        self._on_probe_color_changed(self.probe_panel.color_select.currentText())

    def _on_channel_config_changed(self):
        """Handle channel config changes (DMX address or unit type)."""
        # Update probe panel when selected color's unit changes
        self._on_probe_color_changed(self.probe_panel.color_select.currentText())

    def _on_probe_color_changed(self, color_name: str):
        """Update probe panel when color selection changes."""
        if not color_name:
            return
        config = self.channel_config.get_config(color_name)
        unit_type = config.get("unit_type", "")
        self.probe_panel.set_unit(unit_type)

        # Also load any saved values for this color
        m = self.color_dmx_map.get(color_name, {}) or {}
        self.probe_panel.set_from_channel_map(m)

    # ---------- Beat sync ----------

    def _beat_refresh(self):
        current_lat = self.beat_panel.latency_ms()
        self.detector.set_latency_ms(current_lat)
        st = self.detector.status()
        self.beat_panel.set_bpm(st["bpm"])
        self.beat_panel.set_conf(st["confidence"])
        # keep last latency value for delta computation if nothing else updated it
        self._last_latency_ms = current_lat


    def _next_beat_time_from_phase(self, bpm: float, last_beat_time: float, latency_ms: int) -> float | None:
        if bpm <= 0 or last_beat_time <= 0:
            return None
        period = 60.0 / bpm
        now = time.time()
        n = max(0, math.floor((now - last_beat_time) / period))
        return last_beat_time + (n + 1) * period + (latency_ms / 1000.0)

    def _beat_sync_now(self):
        payload = self.detector.build_sync_payload()
        if not payload:
            self._set_status("Beat Sync: no BPM yet.")
            return
        bpm = payload["bpm"]
        tnext = payload["next_beat_time"]
        delay_ms = max(0, int((tnext - time.time()) * 1000))
        self.seq.stop()
        self.seq.set_bpm(bpm)
        QtCore.QTimer.singleShot(delay_ms, self._aligned_start)

    def _on_auto_sync_toggled(self, enabled: bool):
        """Start or stop the continuous auto-sync timer."""
        if enabled:
            # Do an initial hard sync, then start tracking
            self._beat_sync_now()
            self._auto_sync_timer.start()
            self._set_status("Auto Sync: ON (continuous).")
        else:
            self._auto_sync_timer.stop()
            self._set_status("Auto Sync: OFF.")

    def _auto_sync_tick(self):
        """Continuously track BPM and nudge phase toward detected beat."""
        if not self.beat_panel.is_auto_sync_on():
            return
        if not self.seq.playing:
            return

        payload = self.detector.build_sync_payload()
        if not payload:
            return

        st = self.detector.status()
        # Only trust reasonably confident detections
        if st["confidence"] < 0.03:
            return

        # --- Smoothly track BPM ---
        new_bpm = payload["bpm"]
        bpm_diff = new_bpm - self.seq.bpm
        if abs(bpm_diff) > 0.3:
            # Apply small correction to avoid jerkiness (20% of error per tick)
            corrected = self.seq.bpm + 0.2 * bpm_diff
            self.seq.set_bpm(corrected)
            # Update slider without re-triggering BPM set
            self.bpm_slider.blockSignals(True)
            self.bpm_slider.setValue(int(round(corrected)))
            self.bpm_slider.blockSignals(False)
            self._sync_bpm_label(corrected)

        # --- Phase correction ---
        # Convert detector's next_beat_time (wall time) to perf_counter
        tnext_beat_wall = payload["next_beat_time"]
        perf_wall_offset = time.perf_counter() - time.time()
        tnext_beat_perf = tnext_beat_wall + perf_wall_offset

        # Find sequencer's next "beat" (next even step, since steps are eighth notes)
        step_interval = 60.0 / self.seq.bpm / 2.0
        next_step_at = self.seq._next_tick_at
        if next_step_at is None:
            return
        next_step_idx = (self.seq.step + 1) % self.seq.beats
        # If next step is even, that's a beat; otherwise the one after is
        if next_step_idx % 2 == 0:
            seq_next_beat_perf = next_step_at
        else:
            seq_next_beat_perf = next_step_at + step_interval

        # Compute phase error, wrapped to [-beat/2, +beat/2]
        beat_period = 60.0 / self.seq.bpm
        error_s = tnext_beat_perf - seq_next_beat_perf
        while error_s > beat_period / 2:
            error_s -= beat_period
        while error_s < -beat_period / 2:
            error_s += beat_period
        error_ms = error_s * 1000.0

        # Nudge gently: 25% of error, capped, and ignore tiny errors
        if abs(error_ms) > 3 and abs(error_ms) < 200:
            nudge = 0.25 * error_ms
            self.seq.nudge_phase_ms(nudge)

    def _on_latency_changed_live(self, _v: int):
        """
        LIVE: adjust phase without restarting the bar.
        We compute the delta latency and nudge the sequencer's next tick timing.
        """
        new_lat = self.beat_panel.latency_ms()
        delta = int(new_lat) - int(getattr(self, "_last_latency_ms", 0))

        # update detector immediately (so Sync Now uses the new latency)
        self.detector.set_latency_ms(new_lat)

        if self.seq.playing and delta != 0:
            # Negative delta = advance (next tick sooner); Positive delta = delay next tick
            self.seq.nudge_phase_ms(delta)

        # remember last applied
        self._last_latency_ms = new_lat



    def _aligned_start(self):
        self.seq.step = -1
        self.seq.start()
        self._set_status("Sequencer phase-aligned to next beat.")

    # ---------- UI helpers ----------
    def _on_steps_changed(self, steps):
        """Handle step count change from spinbox."""
        was_playing = self.seq.playing
        if was_playing:
            self.seq.stop()
        self.seq.set_beats(steps)
        self.grid.update_step_count()
        if was_playing:
            self.seq.step = -1
            self.seq.start()

    def _on_clear_all(self):
        """Zero out all cells in the pattern grid."""
        self.seq.pattern[:] = 0
        self.grid.refresh_all()
        self._set_status("Pattern cleared.")

    def _open_channel_config_dialog(self):
        """Show the Channel Configuration dialog (non-modal, brought to front)."""
        self.channel_config_dialog.show()
        self.channel_config_dialog.raise_()
        self.channel_config_dialog.activateWindow()

    def _open_kick_tuning_dialog(self):
        """Show the kick detector tuning dialog (lazily created, non-modal)."""
        if self._kick_tuning_dialog is None:
            self._kick_tuning_dialog = KickTuningDialog(
                self.detector,
                get_offset_ms=lambda: self._kick_paint_offset_ms,
                set_offset_ms=self._set_kick_paint_offset,
                parent=self,
            )
        self._kick_tuning_dialog.show()
        self._kick_tuning_dialog.raise_()
        self._kick_tuning_dialog.activateWindow()

    def _set_kick_paint_offset(self, ms: int):
        self._kick_paint_offset_ms = int(ms)

    def _paint_current_step(self, idx: int):
        """Paint the pattern cell at the current sequencer step for channel idx,
        and refresh that cell in the grid."""
        step = self.seq.step
        if step < 0 or step >= self.seq.beats:
            return
        self.seq.pattern[idx, step] = 1
        # Repaint the affected grid cell (with highlight since it's the current step)
        row_offset, col = self.grid._step_to_grid(step)
        self.grid._paint(row_offset + idx, col, True)

    def _on_paint_pressed(self, idx: int):
        """User is holding down a paint button: light up the channel,
        and arm painting so each step the sequencer advances to while held
        gets written into the pattern (handled in on_step)."""
        self._paint_active[idx] = True

        # Light up the channel right now (visual feedback while held)
        self.active_colors[idx] = 1
        step_sec = self.seq.step_interval_seconds(self.seq.bpm)
        gates = self.gate_panel.gates()
        self.gate_timers[idx].stop()
        gate_ms = max(5, int(step_sec * (gates[idx] / 100.0) * 1000.0))
        self.gate_timers[idx].start(gate_ms)
        self.preview.set_values(self.active_colors)
        self._dmx_push_from_active()

        # If the sequencer is stopped, paint the currently-highlighted step once
        # (otherwise on_step will take over at the next tick)
        if not self.seq.playing:
            self._paint_current_step(idx)

    def _on_paint_released(self, idx: int):
        """Release the paint button: stop painting future steps."""
        self._paint_active[idx] = False

    # ---------- Auto-paint on kick ----------
    def _kick_paint_tick(self):
        """Poll the kick detector and pulse-paint enabled channels on each new onset."""
        if self._closing or self._probe_active:
            return
        current = self.detector.kick.onset_count
        if current == self._last_kick_count:
            return
        # One onset per tick is enough; don't bulk-paint if we missed several
        self._last_kick_count = current
        active_idxs = [i for i, chk in enumerate(self.kick_chks) if chk.isChecked()]
        if not active_idxs:
            return
        # Determine target step, optionally shifted back by the latency offset
        step = self.seq.step
        if step < 0:
            return
        if self._kick_paint_offset_ms != 0 and self.seq.playing and self.seq.bpm > 0:
            step_int_ms = 60000.0 / self.seq.bpm / 2.0
            steps_back = int(round(self._kick_paint_offset_ms / step_int_ms))
            step = (step - steps_back) % self.seq.beats
        for idx in active_idxs:
            self._kick_paint_pulse(idx, step)

    def _kick_paint_pulse(self, idx: int, target_step: int):
        """Paint one cell in the pattern and briefly flash the light for the channel."""
        if not (0 <= target_step < self.seq.beats):
            return
        # Write pattern
        self.seq.pattern[idx, target_step] = 1
        # Refresh the grid cell (highlighted if it IS the current step)
        row_offset, col = self.grid._step_to_grid(target_step)
        self.grid._paint(row_offset + idx, col, target_step == self.seq.step)
        # Flash the light
        self.active_colors[idx] = 1
        self.preview.set_values(self.active_colors)
        step_sec = self.seq.step_interval_seconds(self.seq.bpm) if self.seq.bpm > 0 else 0.2
        gates = self.gate_panel.gates()
        gate_ms = max(5, int(step_sec * (gates[idx] / 100.0) * 1000.0))
        self.gate_timers[idx].stop()
        self.gate_timers[idx].start(gate_ms)
        self._dmx_push_from_active()

    def _sync_bpm_label(self, bpm):
        self.bpm_label.setText(f"{int(bpm)} BPM")

    def _set_status(self, text):
        self.status_lbl.setText(text)

    # ---------- Step & Gate ----------
    @QtCore.pyqtSlot(int, list)
    def on_step(self, step_idx, values):
        if self._closing or self._probe_active:
            return

        # Paint mode: if any channel button is held, mark this step as ON
        # for that channel in the pattern and force it active for this step.
        if any(self._paint_active):
            for row_idx, painting in enumerate(self._paint_active):
                if painting and 0 <= step_idx < self.seq.beats:
                    self.seq.pattern[row_idx, step_idx] = 1
                    values[row_idx] = 1

        self.grid.set_step_highlight(step_idx)
        self.preview.set_values(values)
        for t in self.gate_timers:
            t.stop()
        step_sec = self.seq.step_interval_seconds(self.seq.bpm)
        gates = self.gate_panel.gates()
        for row_idx, active in enumerate(values):
            if active:
                gate_ms = max(5, int(step_sec * (gates[row_idx] / 100.0) * 1000.0))
                self.gate_timers[row_idx].start(gate_ms)
        self.active_colors = values[:]
        self._dmx_push_from_active()
        self.last_active_colors = values[:]

    def _on_gate_timeout(self, row_idx):
        if self._closing:
            return
        self.preview.clear_channel(row_idx)
        self.active_colors[row_idx] = 0
        self._dmx_push_from_active()

    # ---------- DMX merge (v2.6: uses per-color DMX addresses) ----------
    def _dmx_push_from_active(self):
        if self._closing or self._probe_active or not self.dmx.enabled:
            return

        # Get channel config for computing absolute DMX addresses
        channel_configs = self.channel_config.get_all_config()

        # Track all controlled channels across all colors
        controlled = set()
        for c in self.color_names:
            cfg = channel_configs.get(c, {})
            base_addr = cfg.get("dmx_addr", 1)
            m = self.color_dmx_map.get(c, {})
            for rel_ch in m.keys():
                abs_ch = base_addr + int(rel_ch) - 1  # Convert relative to absolute
                controlled.add(abs_ch)

        # Merge active colors
        final = {}
        for idx, active in enumerate(self.active_colors):
            if not active:
                continue
            c = self.color_names[idx]
            cfg = channel_configs.get(c, {})
            base_addr = cfg.get("dmx_addr", 1)
            m = self.color_dmx_map.get(c, {})
            for rel_ch, val in m.items():
                abs_ch = base_addr + int(rel_ch) - 1  # Convert relative to absolute
                if abs_ch not in final:
                    final[abs_ch] = int(val)

        # Build pairs: set controlled channels to their value or 0
        pairs = [(ch, final.get(ch, 0)) for ch in controlled]
        if pairs:
            self.dmx.update_channels(pairs)

    # ---------- Probe (v2.6: uses per-color DMX addresses) ----------
    def _on_probe_start(self, vals_by_channel: dict):
        """Start test mode - send values directly to DMX."""
        color = self.probe_panel.color_select.currentText()
        cfg = self.channel_config.get_config(color)
        base_addr = cfg.get("dmx_addr", 1)

        # Convert relative channels to absolute
        pairs = []
        for rel_ch, val in vals_by_channel.items():
            abs_ch = base_addr + int(rel_ch) - 1
            if int(val) > 0:
                pairs.append((abs_ch, int(val)))

        self._last_probe_channels = [ch for ch, _ in pairs]
        self._probe_active = True
        if pairs:
            self.dmx.update_channels(pairs)
            self._set_status(f"Probe START: {len(pairs)} ch set (base addr {base_addr})")

    def _on_probe_update(self, vals_by_channel: dict):
        if not self._probe_active:
            return

        color = self.probe_panel.color_select.currentText()
        cfg = self.channel_config.get_config(color)
        base_addr = cfg.get("dmx_addr", 1)

        pairs = []
        for rel_ch, val in vals_by_channel.items():
            abs_ch = base_addr + int(rel_ch) - 1
            pairs.append((abs_ch, int(val)))

        touched = [ch for ch, _ in pairs]
        self._last_probe_channels = sorted(set(self._last_probe_channels + touched))
        if pairs:
            self.dmx.update_channels(pairs)

    def _on_probe_stop(self, vals_by_channel: dict):
        if self._last_probe_channels:
            self.dmx.update_channels([(ch, 0) for ch in self._last_probe_channels])
        self._last_probe_channels = []
        self._probe_active = False
        self._dmx_push_from_active()
        self._set_status("Probe STOP")

    def _on_probe_write(self, color_name: str, vals_by_channel: dict):
        """Write current probe values to a color's DMX map (stores relative channels)."""
        ch_map = {}
        for ch, v in vals_by_channel.items():
            if int(v) > 0:
                ch_map[int(ch)] = int(v)
        self.color_dmx_map[color_name] = dict(ch_map)
        self._set_status(f"Wrote {len(ch_map)} channels to '{color_name}'")

    def _on_probe_read(self, color_name: str):
        m = self.color_dmx_map.get(color_name, {}) or {}
        self.probe_panel.set_from_channel_map(m)
        self._set_status(f"Read {len(m)} channels from '{color_name}'")


    def _on_play_pressed(self):
        """Immediate restart from step 0 (bar 1)."""

        # Cancel any pending align timers (sync/latency actions)
        self._cancel_align_timer()

        # Stop current sequence
        self.seq.stop()

        # Reset step so next emitted is 0
        self.seq.step = -1

        # Ensure next tick isn't "already due"
        self.seq._next_tick_at = time.perf_counter() + 0.001

        # Clear gate visuals
        try:
            for t in self.gate_timers:
                t.stop()
            self.preview.clear_values()
            self.active_colors = [0] * len(self.color_names)
            self.last_active_colors = [0] * len(self.color_names)
        except Exception:
            pass

        # Start playing from tick 0
        self.seq.start()
        self._set_status("Restarted sequence from bar 1.")



    def _on_stop_pressed(self):
        """Stop now and cancel any scheduled phase-alignment."""
        # cancel any pending align-to-next-beat start
        self._cancel_align_timer()
        # stop sequencer
        self.seq.stop()
        # optional: visually clear current step/gates
        try:
            for t in self.gate_timers:
                t.stop()
            self.preview.clear_values()
            self.active_colors = [0] * len(self.color_names)
            self.last_active_colors = [0] * len(self.color_names)
        except Exception:
            pass
        # Zero out DMX channels so lights turn off
        self._dmx_push_from_active()
        self._set_status("Stopped.")

    # ---------- DMX config ----------
    def _apply_dmx_config(self):
        cfg = self.dmx_panel.config()
        self.dmx.set_url(cfg["url"])
        self.dmx.set_enabled(cfg["enabled"])


    def _cancel_align_timer(self):
        """Cancel any pending alignment timer."""
        if getattr(self, "_align_timer", None):
            try:
                self._align_timer.stop()
                self._align_timer.deleteLater()
            except Exception:
                pass
            self._align_timer = None

    def _schedule_align(self, delay_ms: int, preserve_bpm: bool = True, bpm_value: float | None = None):
        """
        Schedule a start exactly on the next beat after delay_ms.
        - If preserve_bpm=True, do NOT change current seq.bpm.
        - If preserve_bpm=False and bpm_value is provided, set seq.bpm=bpm_value before starting.
        Always clears any previous pending align timer first.
        """
        self._cancel_align_timer()

        # Prepare the start action
        def _do_start():
            # Make sure we are not in closing state
            if self._closing:
                return
            # Optional BPM update
            if not preserve_bpm and bpm_value is not None:
                self.seq.set_bpm(float(bpm_value))
            # Reset bar and start
            self.seq.step = -1
            self.seq.start()
            # clear this timer ref
            self._align_timer = None
            self._set_status("Sequencer phase-aligned to next beat.")

        # QTimer.singleShot loses handle; use a QTimer instance so we can cancel it
        self._align_timer = QtCore.QTimer(self)
        self._align_timer.setSingleShot(True)
        self._align_timer.timeout.connect(_do_start)
        self._align_timer.start(max(0, int(delay_ms)))


    # ---------- Slots: FULL SCENE save/load (v2.6: includes channel config) ----------
    def _collect_slot_state(self):
        return {
            "pattern": self.seq.pattern.tolist(),
            "beats":   self.seq.beats,
            "gates":   self.gate_panel.gates(),
            "dmx":     self.dmx_panel.config(),
            "color_dmx_map": self.color_dmx_map,
            "channel_config": self.channel_config.get_all_config(),  # v2.6: per-color config
            # BPM intentionally NOT stored in slot
        }

    def _coerce_pattern(self, patt_list, target_cols=None):
        """Accept arbitrary list/array; coerce to (rows=len(channels), cols=target_cols).
           target_cols defaults to current seq.beats."""
        if target_cols is None:
            target_cols = self.seq.beats
        rows = len(self.color_names)
        try:
            arr = np.array(patt_list, dtype=np.uint8)
        except Exception:
            return np.zeros((rows, target_cols), dtype=np.uint8)
        if arr.ndim != 2:
            return np.zeros((rows, target_cols), dtype=np.uint8)
        # Fix row count
        if arr.shape[0] != rows:
            out = np.zeros((rows, arr.shape[1]), dtype=np.uint8)
            rmin = min(rows, arr.shape[0])
            out[:rmin, :] = arr[:rmin, :]
            arr = out
        # Fix col count
        if arr.shape[1] == target_cols:
            return arr
        if arr.shape[1] > target_cols:
            return arr[:, :target_cols]
        # pad to right
        out = np.zeros((rows, target_cols), dtype=np.uint8)
        out[:, :arr.shape[1]] = arr
        return out

    def _apply_slot_state(self, state: dict):
        """Apply full scene EXCEPT BPM/phase."""
        if not state:
            return

        # Restore beat count (default 16 for backwards compat)
        beats = state.get("beats", 16)
        beats = max(4, min(16, int(beats)))
        self.seq.set_beats(beats)
        self.steps_spin.setValue(beats)

        # Pattern
        patt = self._coerce_pattern(state.get("pattern", []), beats)
        if patt.shape == (len(self.color_names), beats):
            self.seq.pattern = patt
            self.grid.update_step_count()

        # Gates
        gates = state.get("gates")
        if gates and len(gates) == len(self.color_names):
            self.gate_panel.set_gates(gates)

        # DMX output config
        dmx_cfg = state.get("dmx")
        if dmx_cfg:
            self.dmx_panel.set_config(dmx_cfg)
            self._apply_dmx_config()

        # Color DMX personalities (relative channel maps)
        raw_map = state.get("color_dmx_map", {})
        fixed = {}
        for color, mapping in (raw_map or {}).items():
            fixed[color] = {int(k): int(v) for k, v in (mapping or {}).items()}
        self.color_dmx_map = fixed

        # v2.6: Channel config (DMX addresses and unit types)
        ch_cfg = state.get("channel_config", {})
        if ch_cfg:
            self.channel_config.set_all_config(ch_cfg)

        # Refresh probe with selected color
        cur = self.probe_panel.color_select.currentText()
        self._on_probe_color_changed(cur)

        # clear active indicators
        self.active_colors = [0] * len(self.color_names)
        self.last_active_colors = [0] * len(self.color_names)

        self._set_status("Scene loaded from slot.")

    def save_current_slot(self):
        slot_id = self.slots_panel.btn_group.checkedId() or 1
        self.slots[str(slot_id)] = self._collect_slot_state()
        self._set_status(f"Saved scene to slot {slot_id} (in memory). Use 'Save Config File…' to persist.")

    def load_current_slot(self):
        slot_id = self.slots_panel.btn_group.checkedId() or 1
        state = self.slots.get(str(slot_id))
        if not state:
            QtWidgets.QMessageBox.information(self, "Empty slot", f"Slot {slot_id} is empty (no scene in memory).")
            self._set_status(f"Slot {slot_id} is empty.")
            return
        self._apply_slot_state(state)
        QtWidgets.QMessageBox.information(self, "Loaded", f"Scene loaded from slot {slot_id}.")

    # ---------- Unified config (v2.6: includes channel config) ----------
    def _collect_global(self):
        return {
            "dmx_config": self.dmx_panel.config(),
            "bpm": int(self.seq.bpm),
            "beats": self.seq.beats,
            "latency_ms": int(self.beat_panel.latency_ms()),
            "gate_values": [float(s.value()) for s in self.gate_panel.sliders],
            "color_dmx_map": self.color_dmx_map,
            "channel_config": self.channel_config.get_all_config(),  # v2.6
        }

    def _apply_global(self, g: dict):
        if not g:
            return
        dmx = g.get("dmx_config")
        if dmx:
            self.dmx_panel.set_config(dmx)
            self._apply_dmx_config()
        beats = g.get("beats")
        if beats is not None:
            beats = max(4, min(16, int(beats)))
            self.seq.set_beats(beats)
            self.steps_spin.setValue(beats)
            self.grid.update_step_count()
        bpm = g.get("bpm")
        if bpm is not None:
            self.bpm_slider.setValue(int(bpm))
        lat = g.get("latency_ms")
        if lat is not None:
            self.beat_panel.lat_sld.setValue(int(lat))
        gates = g.get("gate_values")
        if gates and len(gates) == len(self.color_names):
            self.gate_panel.set_gates(gates)
        cm = g.get("color_dmx_map", {})
        fixed = {}
        for color, m in (cm or {}).items():
            fixed[color] = {int(k): int(v) for k, v in (m or {}).items()}
        self.color_dmx_map = fixed

        # v2.6: Channel config
        ch_cfg = g.get("channel_config", {})
        if ch_cfg:
            self.channel_config.set_all_config(ch_cfg)

        cur = self.probe_panel.color_select.currentText()
        self._on_probe_color_changed(cur)

    def save_full_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Complete Config", CONFIG_FILE, "JSON Files (*.json)")
        if not path:
            return
        data = {"version": 4, "global": self._collect_global(), "slots": self.slots}  # v2.6 = version 4
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            QtWidgets.QMessageBox.information(self, "Success", "Full configuration saved.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not save:\n{e}")

    def load_full_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Complete Config", CONFIG_FILE, "JSON Files (*.json)")
        if not path:
            return
        self._load_config_path(path)

    def _load_config_path(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not load file:\n{e}")
            return
        try:
            self._apply_global(data.get("global", {}))
            disk_slots = data.get("slots", {})
            self.slots = {str(n): disk_slots.get(str(n)) for n in SLOTS}
            # auto-select & load slot 1 if present
            self.slots_panel.btn_group.button(1).setChecked(True)
            if self.slots["1"]:
                self._apply_slot_state(self.slots["1"])
            self._set_status(f"Loaded config from {os.path.basename(path)} (v2.6)")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not apply configuration:\n{e}")

    # ---------- Shutdown ----------
    def cleanup(self):
        if self._closing:
            return
        self._closing = True

        # Set global shutdown flag FIRST to break all spin-wait loops immediately
        _shutdown_flag.set()

        # Disconnect signals first to prevent callbacks during shutdown
        try:
            self.seq.step_changed.disconnect(self.on_step)
        except Exception:
            pass

        try:
            self.seq.tempo_changed.disconnect()
        except Exception:
            pass

        # Stop beat detection timer
        try:
            self.beat_timer.stop()
        except Exception:
            pass

        # Stop auto-sync timer
        try:
            self._auto_sync_timer.stop()
        except Exception:
            pass

        # Stop kick-paint timer
        try:
            self._kick_paint_timer.stop()
        except Exception:
            pass

        # Cancel any pending alignment timers
        try:
            self._cancel_align_timer()
        except Exception:
            pass

        # Stop gate timers
        try:
            for t in self.gate_timers:
                try:
                    t.stop()
                except Exception:
                    pass
        except Exception:
            pass

        # Stop audio detector BEFORE sequencer (audio callback might trigger processing)
        try:
            self.detector.stop()
        except Exception:
            pass

        # Stop sequencer
        try:
            self.seq.stop()
        except Exception:
            pass

        # Clear DMX channels before stopping
        try:
            if self._probe_active and self._last_probe_channels:
                self.dmx.update_channels([(ch, 0) for ch in self._last_probe_channels])
        except Exception:
            pass
        self._probe_active = False
        self._last_probe_channels = []

        # Stop DMX
        try:
            self.dmx.stop()
        except Exception:
            pass

    # ---------- Keyboard shortcuts for paint buttons (1/2/3/4) ----------
    _PAINT_KEY_MAP = {
        QtCore.Qt.Key_1: 0,
        QtCore.Qt.Key_2: 1,
        QtCore.Qt.Key_3: 2,
        QtCore.Qt.Key_4: 3,
    }

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if e.isAutoRepeat():
            return super().keyPressEvent(e)
        idx = self._PAINT_KEY_MAP.get(e.key())
        if idx is not None and idx < len(self.paint_btns):
            # Visually press the button and trigger the paint
            self.paint_btns[idx].setDown(True)
            self._on_paint_pressed(idx)
            e.accept()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QtGui.QKeyEvent):
        if e.isAutoRepeat():
            return super().keyReleaseEvent(e)
        idx = self._PAINT_KEY_MAP.get(e.key())
        if idx is not None and idx < len(self.paint_btns):
            self.paint_btns[idx].setDown(False)
            self._on_paint_released(idx)
            e.accept()
            return
        super().keyReleaseEvent(e)

    def closeEvent(self, e):
        try:
            self.cleanup()
        except Exception:
            pass
        # Always accept the close event
        e.accept()
        # Force quit the application
        try:
            QtWidgets.QApplication.instance().quit()
        except Exception:
            pass


# ---------------- main ----------------

def _force_shutdown():
    """Emergency shutdown - set flag and force exit if threads don't stop."""
    _shutdown_flag.set()

def main():
    # Load unit definitions from file
    global UNIT_DEFINITIONS
    UNIT_DEFINITIONS = load_units_from_file(UNITS_FILE)
    print(f"Loaded {len(UNIT_DEFINITIONS)} unit definitions from {UNITS_FILE}")

    # Register emergency shutdown handler early
    atexit.register(_force_shutdown)

    # Set shutdown flag on exit
    _shutdown_flag.clear()

    app = QtWidgets.QApplication(sys.argv)

    # Ensure app quits when last window closes
    app.setQuitOnLastWindowClosed(True)

    w = MainWindow()
    w.resize(850, 820)  # Slightly larger to accommodate new panel
    w.show()

    exit_code = 0
    try:
        exit_code = app.exec()
    except Exception:
        pass
    finally:
        # Ensure shutdown flag is set
        _shutdown_flag.set()
        # Run cleanup
        try:
            w.cleanup()
        except Exception:
            pass
        # Give threads a brief moment to exit
        time.sleep(0.05)

    return exit_code


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        exit_code = 1
    finally:
        # Ensure shutdown flag is set
        _shutdown_flag.set()
    # Force immediate exit - don't wait for threads
    os._exit(exit_code)
