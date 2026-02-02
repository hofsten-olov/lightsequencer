
# beat_sync.py
# Mic → BPM via spectral-flux onsets + autocorrelation (pure NumPy), no aubio needed.
# Provides BeatDetector class with: start(), stop(), status(), set_latency_ms(int SIGNED), build_sync_payload()

import time, math, threading, collections
import numpy as np
import sounddevice as sd

def _hamming(N: int) -> np.ndarray:
    n = np.arange(N)
    return (0.54 - 0.46*np.cos(2*np.pi*n/(N-1))).astype(np.float32)

class _FluxTempo:
    """
    Real-time spectral-flux onset detection + BPM estimation (autocorrelation).
    Tuned for 44.1 kHz / 1024 frame / 512 hop (~86 fps). Adjust sr/frame/hop as needed.
    """
    def __init__(self, sr=44100, frame=1024, hop=512,
                 flux_threshold_k=1.5, ema_alpha=0.12,
                 bpm_min=60, bpm_max=180):
        self.sr = sr; self.frame = frame; self.hop = hop
        self.win = _hamming(frame)
        self._prev_mag = None
        self._overlap = np.zeros((frame-hop,), dtype=np.float32)
        self.flux_hist = collections.deque(maxlen=200)
        self.onset_times = collections.deque(maxlen=64)
        self.k = flux_threshold_k
        self.alpha = ema_alpha
        self._ema_init = False
        self.bpm = 0.0
        self.confidence = 0.0
        self.last_beat_time = 0.0
        self.bpm_min = bpm_min; self.bpm_max = bpm_max

    def _frames(self, x_mono: np.ndarray):
        data = np.concatenate([self._overlap, x_mono])
        i = 0
        while i + self.frame <= len(data):
            yield data[i:i+self.frame]
            i += self.hop
        tail = data[i:]
        keep = self.frame - self.hop
        self._overlap = tail[-keep:].copy() if len(tail) >= keep else tail.copy()

    def _spectral_flux(self, f: np.ndarray) -> float:
        xw = f * self.win
        mag = np.abs(np.fft.rfft(xw))
        if self._prev_mag is None:
            self._prev_mag = mag
            return 0.0
        diff = mag - self._prev_mag
        self._prev_mag = mag
        return float(np.sum(np.clip(diff, 0, None)))

    def _maybe_onset(self, flux: float, now: float) -> bool:
        self.flux_hist.append(flux)
        if len(self.flux_hist) < 20:
            return False
        mu = np.mean(self.flux_hist)
        sd = np.std(self.flux_hist) + 1e-6
        thr = mu + self.k*sd
        if flux > thr:
            # debounce 80 ms
            if len(self.onset_times) == 0 or (now - self.onset_times[-1]) > 0.08:
                self.onset_times.append(now)
                return True
        return False

    def _estimate_bpm(self) -> tuple[float, float]:
        if len(self.onset_times) < 6:
            return 0.0, 0.0
        tt = np.array(self.onset_times)
        # build binary onset train @ 100 Hz over last ~8 s
        sr_env = 100.0; T = 8.0
        now = tt[-1]; t0 = now - T
        n = max(50, int(T*sr_env))
        env = np.zeros(n, dtype=np.float32)
        for t in tt:
            if t >= t0:
                idx = int((t - t0)*sr_env)
                if 0 <= idx < n: env[idx] = 1.0
        ac = np.correlate(env - env.mean(), env - env.mean(), mode='full')
        ac = ac[ac.size//2:]
        min_lag = int(sr_env * 60.0 / self.bpm_max)
        max_lag = int(sr_env * 60.0 / self.bpm_min)
        min_lag = max(1, min_lag); max_lag = min(len(ac)-1, max_lag)
        if max_lag <= min_lag: return 0.0, 0.0
        roi = ac[min_lag:max_lag]
        if roi.size < 3: return 0.0, 0.0
        k = int(np.argmax(roi)) + min_lag
        # parabolic refinement
        if 1 <= k < len(ac)-1:
            y0,y1,y2 = ac[k-1],ac[k],ac[k+1]
            denom = 2*(y0 - 2*y1 + y2)
            off = (y0 - y2)/denom if abs(denom) > 1e-6 else 0.0
        else:
            off = 0.0
        lag = k + off
        period = lag / sr_env
        bpm = 60.0 / max(period, 1e-6)
        conf = float(roi.max() / (np.sum(roi) + 1e-6))
        return bpm, conf

    def process_block(self, mono: np.ndarray):
        now = time.time()
        onset = False
        for fr in self._frames(mono):
            flux = self._spectral_flux(fr)
            onset |= self._maybe_onset(flux, now)
        if onset:
            est, conf = self._estimate_bpm()
            if est > 0:
                if self.bpm > 0:
                    est = 0.7*self.bpm + 0.3*est  # suppress jumps
                if not self._ema_init:
                    self.bpm = est; self._ema_init = True
                else:
                    self.bpm = (1.0 - self.alpha)*self.bpm + self.alpha*est
                self.confidence = conf
                self.last_beat_time = now


class BeatDetector:
    """
    High-level wrapper using sounddevice InputStream.
    Methods:
        start(device=None), stop(), set_latency_ms(int SIGNED),
        status() -> dict, build_sync_payload() -> dict|None
    """
    def __init__(self, samplerate=44100, bufsize=1024, hopsize=512,
                 bpm_min=60, bpm_max=180, ema_alpha=0.12):
        self.sr = samplerate; self.bufsize = bufsize; self.hopsize = hopsize
        self.proc = _FluxTempo(sr=samplerate, frame=bufsize, hop=hopsize,
                               ema_alpha=ema_alpha, bpm_min=bpm_min, bpm_max=bpm_max)
        self._stream = None
        self._running = False
        self._stopping = False
        self._lock = threading.Lock()
        self._latency_ms = 0  # signed

    def _callback(self, indata, frames, time_info, status):
        # Quick exit if not running or stopping
        if not self._running or self._stopping:
            return
        try:
            mono = np.mean(indata, axis=1).astype(np.float32)
            self.proc.process_block(mono)
        except Exception:
            pass  # Ignore errors during callback

    def start(self, device=None):
        if self._running:
            return
        self._running = True
        self._stopping = False
        try:
            self._stream = sd.InputStream(samplerate=self.sr, blocksize=self.bufsize,
                                          channels=1, dtype='float32', device=device,
                                          callback=self._callback)
            self._stream.start()
        except Exception:
            self._running = False
            self._stream = None

    def stop(self):
        # Set stopping flag first to make callback exit immediately
        self._stopping = True
        self._running = False

        stream = self._stream
        self._stream = None

        if stream is not None:
            try:
                # Use abort() instead of stop() for faster termination
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def set_latency_ms(self, ms: int):
        with self._lock:
            # allow negative and positive offsets
            self._latency_ms = int(ms)

    def status(self) -> dict:
        return {
            "bpm": float(self.proc.bpm),
            "confidence": float(self.proc.confidence),
            "last_beat_time": float(self.proc.last_beat_time),
            "latency_ms": int(self._latency_ms),
        }

    def build_sync_payload(self) -> dict | None:
        st = self.status()
        bpm = st["bpm"]; last = st["last_beat_time"]
        if bpm <= 0 or last <= 0: return None
        period = 60.0 / bpm
        now = time.time()
        n = max(0, math.floor((now - last)/period))
        # next beat based on detected phase + signed latency adjustment
        tnext = last + (n + 1)*period + (self._latency_ms/1000.0)
        return {
            "bpm": round(bpm, 2),
            "last_beat_time": last,
            "next_beat_time": tnext,
            "latency_ms": int(self._latency_ms),
            "generated_at": now
        }
