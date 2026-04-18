#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Beat Sync Tuner — standalone GUI for tuning beat detection parameters.
Imports beat_sync.py, runs the mic listener, and lets you tweak every
parameter live via sliders. Includes waveform display, level meter,
onset flash, and audio device selector.
"""

import sys, time
import numpy as np
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore, QtGui
from beat_sync import BeatDetector


class WaveformWidget(QtWidgets.QWidget):
    """Live waveform display showing recent audio from the mic."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self._samples = np.zeros(1024, dtype=np.float32)

    def set_samples(self, samples):
        self._samples = samples
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2

        # Background
        p.fillRect(0, 0, w, h, QtGui.QColor(25, 25, 30))

        # Center line
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 70), 1))
        p.drawLine(0, int(mid), w, int(mid))

        # Waveform
        n = len(self._samples)
        if n < 2:
            return
        pen = QtGui.QPen(QtGui.QColor(80, 200, 120), 1.5)
        p.setPen(pen)

        step = max(1, n // w)
        path = QtGui.QPainterPath()
        first = True
        for i in range(0, min(n, w * step), step):
            x = (i / max(1, n - 1)) * w
            y = mid - self._samples[i] * mid * 0.9
            if first:
                path.moveTo(x, y)
                first = False
            else:
                path.lineTo(x, y)
        p.drawPath(path)


class LevelMeter(QtWidgets.QWidget):
    """Vertical RMS level meter with peak hold."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(24)
        self.setMinimumHeight(80)
        self._level = 0.0
        self._peak = 0.0
        self._peak_time = 0.0

    def set_level(self, rms):
        self._level = min(1.0, rms * 5.0)
        now = time.time()
        if self._level > self._peak:
            self._peak = self._level
            self._peak_time = now
        elif now - self._peak_time > 1.0:
            self._peak = max(self._level, self._peak - 0.02)
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, QtGui.QColor(25, 25, 30))

        bar_h = int(self._level * h)
        for y in range(h - bar_h, h):
            frac = 1.0 - y / h
            if frac > 0.85:
                col = QtGui.QColor(235, 60, 60)
            elif frac > 0.6:
                col = QtGui.QColor(235, 200, 50)
            else:
                col = QtGui.QColor(60, 200, 90)
            p.fillRect(2, y, w - 4, 1, col)

        peak_y = int((1.0 - self._peak) * h)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        p.drawLine(2, peak_y, w - 3, peak_y)


class OnsetFlash(QtWidgets.QWidget):
    """Flashes on detected onsets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        self._on = False
        self._off_timer = QtCore.QTimer(self)
        self._off_timer.setSingleShot(True)
        self._off_timer.timeout.connect(self._turn_off)

    def flash(self):
        self._on = True
        self.update()
        self._off_timer.start(80)

    def _turn_off(self):
        self._on = False
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if self._on:
            p.setBrush(QtGui.QColor(255, 220, 50))
        else:
            p.setBrush(QtGui.QColor(50, 50, 55))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(2, 2, 20, 20)


def _list_input_devices():
    """Return list of (device_index, name) for input-capable devices."""
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            result.append((i, d['name']))
    return result


class BeatTuner(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beat Sync Tuner")
        self.setMinimumWidth(550)

        self.detector = BeatDetector()
        self.proc = self.detector.proc
        self._prev_onset_count = 0

        root = QtWidgets.QVBoxLayout(self)

        # --- Device selector ---
        dev_row = QtWidgets.QHBoxLayout()
        dev_row.addWidget(QtWidgets.QLabel("Input device:"))
        self.dev_combo = QtWidgets.QComboBox()
        self.dev_combo.setMinimumWidth(300)
        self._input_devices = _list_input_devices()
        self.dev_combo.addItem("(default)")
        for idx, name in self._input_devices:
            self.dev_combo.addItem(f"[{idx}] {name}")
        dev_row.addWidget(self.dev_combo, stretch=1)
        self.dev_start_btn = QtWidgets.QPushButton("Start")
        self.dev_start_btn.clicked.connect(self._start_device)
        dev_row.addWidget(self.dev_start_btn)
        root.addLayout(dev_row)

        # --- Audio visualization ---
        vis_row = QtWidgets.QHBoxLayout()
        self.level_meter = LevelMeter()
        self.waveform = WaveformWidget()
        vis_row.addWidget(self.level_meter)
        vis_row.addWidget(self.waveform, stretch=1)
        root.addLayout(vis_row)

        # --- Live readout ---
        readout = QtWidgets.QHBoxLayout()
        self.onset_flash = OnsetFlash()
        self.bpm_lbl = QtWidgets.QLabel("BPM: --")
        self.bpm_lbl.setStyleSheet("font-size: 22px; font-weight: bold;")
        self.conf_lbl = QtWidgets.QLabel("Conf: --")
        self.conf_lbl.setStyleSheet("font-size: 14px;")
        self.onset_lbl = QtWidgets.QLabel("Onsets: 0")
        readout.addWidget(self.onset_flash)
        readout.addWidget(self.bpm_lbl)
        readout.addSpacing(20)
        readout.addWidget(self.conf_lbl)
        readout.addSpacing(20)
        readout.addWidget(self.onset_lbl)
        readout.addStretch()
        root.addLayout(readout)

        root.addWidget(self._separator())

        # --- Parameter sliders ---
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1, 1)
        self._sliders = {}

        params = [
            ("bpm_min",       "BPM min",              60,  160, self.proc.bpm_min,           1, ""),
            ("bpm_max",       "BPM max",             100,  240, self.proc.bpm_max,           1, ""),
            ("threshold_k",   "Onset threshold (k)",   5,   50, int(self.proc.k * 10),     10, ""),
            ("ema_alpha",     "EMA smoothing",         1,   80, int(self.proc.alpha * 100), 100, ""),
            ("bass_cutoff",   "Bass emphasis freq",    50, 2000, int(self.proc.bass_cutoff), 1, "Hz"),
        ]

        for row, (key, label, lo, hi, default, div, unit) in enumerate(params):
            lbl = QtWidgets.QLabel(label)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(lo, hi)
            s.setValue(default)
            val_text = f"{default / div:.2f}" if div > 1 else str(default)
            val_lbl = QtWidgets.QLabel(f"{val_text} {unit}".strip())
            val_lbl.setMinimumWidth(80)
            val_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(lbl, row, 0)
            grid.addWidget(s, row, 1)
            grid.addWidget(val_lbl, row, 2)
            self._sliders[key] = (s, val_lbl, div, unit)
            s.valueChanged.connect(lambda v, k=key: self._on_param_changed(k))

        root.addLayout(grid)

        root.addWidget(self._separator())

        # --- Buttons ---
        btn_row = QtWidgets.QHBoxLayout()
        self.reset_btn = QtWidgets.QPushButton("Reset Detection")
        self.reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self.reset_btn)

        self.print_btn = QtWidgets.QPushButton("Print Current Settings")
        self.print_btn.clicked.connect(self._print_settings)
        btn_row.addWidget(self.print_btn)

        btn_row.addStretch()
        root.addLayout(btn_row)

        # --- Status ---
        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setWordWrap(True)
        root.addWidget(self.status_lbl)

        # Refresh timer
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

        # Auto-start with default device
        self._start_device()

    def _start_device(self):
        """Stop current stream and start with selected device."""
        self.detector.stop()
        self.proc.reset()
        self._prev_onset_count = 0

        combo_idx = self.dev_combo.currentIndex()
        if combo_idx == 0:
            device = None
            dev_name = "default"
        else:
            device = self._input_devices[combo_idx - 1][0]
            dev_name = self._input_devices[combo_idx - 1][1]

        err = self.detector.start(device=device)
        if err:
            self.status_lbl.setText(f"FAILED to open [{dev_name}]: {err}")
            self.status_lbl.setStyleSheet("color: red;")
        else:
            self.status_lbl.setText(f"Listening on: {dev_name}")
            self.status_lbl.setStyleSheet("")

    def _separator(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    def _on_param_changed(self, key):
        s, val_lbl, div, unit = self._sliders[key]
        raw = s.value()
        val = raw / div
        val_text = f"{val:.2f}" if div > 1 else str(raw)
        val_lbl.setText(f"{val_text} {unit}".strip())

        proc = self.proc
        if key == "bass_cutoff":
            proc.set_bass_cutoff(raw)
        elif key == "threshold_k":
            proc.k = val
        elif key == "ema_alpha":
            proc.alpha = val
        elif key == "bpm_min":
            proc.bpm_min = raw
        elif key == "bpm_max":
            proc.bpm_max = raw

    def _refresh(self):
        st = self.detector.status()
        self.bpm_lbl.setText(f"BPM: {st['bpm']:.1f}")
        self.conf_lbl.setText(f"Conf: {st['confidence']:.3f}")
        onset_count = len(self.proc.onset_times)
        self.onset_lbl.setText(f"Onsets: {onset_count}")

        if onset_count > self._prev_onset_count:
            self.onset_flash.flash()
        self._prev_onset_count = onset_count

        self.waveform.set_samples(self.detector._audio_buf.copy())
        self.level_meter.set_level(self.detector._audio_level)

    def _on_reset(self):
        self.proc.reset()
        self._prev_onset_count = 0
        self.status_lbl.setText("Detection reset. Listening...")
        self.status_lbl.setStyleSheet("")

    def _print_settings(self):
        p = self.proc
        lines = [
            "--- Beat Sync Settings ---",
            f"bpm_min      = {p.bpm_min}",
            f"bpm_max      = {p.bpm_max}",
            f"threshold_k  = {p.k:.2f}",
            f"ema_alpha    = {p.alpha:.2f}",
            f"bass_cutoff  = {p.bass_cutoff}",
            "--------------------------",
        ]
        text = "\n".join(lines)
        print(text)
        self.status_lbl.setText("Settings printed to console.")
        self.status_lbl.setStyleSheet("")

    def closeEvent(self, e):
        self._timer.stop()
        self.detector.stop()
        e.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = BeatTuner()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
