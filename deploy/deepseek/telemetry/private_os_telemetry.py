#!/usr/bin/env python3
"""Bounded, read-only native OS samples for one controller-owned private PID.

No HTTP, GPU imports, target mutation, pressure generation, or per-sample
subprocesses. Compile the companion dylib with --prepare before maintenance.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent))
from portable import state_dir
OUTPUT = state_dir() / "telemetry"
NATIVE = HERE / 'private_os_telemetry.c'
LIBRARY = OUTPUT / 'private_os_telemetry.dylib'
ABI_REPORT = OUTPUT / 'private-os-telemetry-abi.json'
HEADERS = ('usr/include/sys/resource.h', 'usr/include/libproc.h',
           'usr/include/mach/vm_statistics.h', 'usr/include/mach/host_info.h',
           'usr/include/mach/mach_host.h', 'usr/include/sys/sysctl.h')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    """CPU-only build and typed SDK ABI evidence; never overwrite evidence."""
    if LIBRARY.exists() or ABI_REPORT.exists():
        raise FileExistsError('Native telemetry already prepared; preserve its source and ABI evidence')
    sdk = Path(subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip())
    command = ['xcrun', 'clang', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
               '-dynamiclib', '-isysroot', str(sdk), str(NATIVE), '-o', str(LIBRARY)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    native = NativeReader(check_provenance=False)
    report = {'status': 'passed', 'cpu_only': True, 'source': {'path': str(NATIVE), 'sha256': sha(NATIVE)},
              'library': {'path': str(LIBRARY), 'sha256': sha(LIBRARY)}, 'compiler': command,
              'sdk': str(sdk), 'headers': [{'path': str(sdk / h), 'sha256': sha(sdk / h)} for h in HEADERS],
              'abi': native.abi, 'created_at': time.time(), 'platform': platform.platform(),
              'interface': 'Only int PID, char output buffer and size_t capacity cross ctypes; structs are SDK typed inside C.',
              'host_fields': 'Every field is admitted only if its end offset is within returned_words*sizeof(integer_t).'}
    assert native.abi['integer_bytes'] == 4
    assert native.abi['rusage_phys_offset'] == 72
    assert native.abi['rusage_resident_offset'] == 64
    assert native.abi['rusage_disk_read_offset'] == 144
    assert native.abi['rusage_start_offset'] == 80
    assert native.abi['vm_decompressions_offset'] == 96
    assert native.abi['vm_compressions_offset'] == 104
    assert native.abi['vm_swapins_offset'] == 112
    assert native.abi['vm_swapouts_offset'] == 120
    assert native.abi['vm_compressor_page_count_offset'] == 128
    assert native.abi['vm_uncompressed_pages_offset'] == 144
    exclusive_json(ABI_REPORT, report)
    return report


class NativeReader:
    def __init__(self, *, check_provenance=True):
        if check_provenance:
            proof = json.loads(ABI_REPORT.read_text())
            if proof['status'] != 'passed' or sha(NATIVE) != proof['source']['sha256'] or sha(LIBRARY) != proof['library']['sha256']:
                raise RuntimeError('Native telemetry source/library provenance changed')
        self.lib = ctypes.CDLL(str(LIBRARY), use_errno=True)
        self.lib.r10_abi_json.argtypes = []
        self.lib.r10_abi_json.restype = ctypes.c_char_p
        self.lib.r10_snapshot_json.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.r10_snapshot_json.restype = ctypes.c_int
        self.abi = json.loads(self.lib.r10_abi_json())
        self.buffer = ctypes.create_string_buffer(16384)

    def __call__(self, pid):
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError('Positive integer PID required')
        rc = self.lib.r10_snapshot_json(pid, self.buffer, len(self.buffer))
        if rc:
            raise OSError(rc, 'Native OS snapshot failed')
        return json.loads(self.buffer.value)


class Sampler:
    """One bounded sampler thread; start after obtaining an owned private PID.

    Use stop(reason) in controller finally BEFORE stopping the owned process.
    Process CPU overhead includes the Python sampling thread only. Native
    counters are observations and never modify or relax the memory guard.
    """
    def __init__(self, pid, report_path, *, interval_s=0.5, max_duration_s=1800,
                 reader=None, provenance=None):
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError('Positive integer PID required')
        if not 0.5 <= interval_s <= 2 or not 0 < max_duration_s <= 1800:
            raise ValueError('Interval must be 0.5..2s and duration must be <=1800s')
        self.pid, self.report_path = pid, Path(report_path)
        self.interval_s, self.max_duration_s = float(interval_s), float(max_duration_s)
        self.reader = reader or NativeReader()
        self.provenance = provenance
        self.samples, self.marks, self.errors = [], [], []
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._lock = threading.Lock()
        self._sample_cpu_ns = 0
        self._sample_wall_ns = 0
        self._max_sample_wall_ns = 0

    def _read(self, kind):
        cpu = time.thread_time_ns(); wall = time.monotonic_ns()
        row = self.reader(self.pid)
        row.update(at=time.time(), monotonic_ns=time.monotonic_ns(), kind=kind)
        process = row['process']
        if not process.get('available'):
            raise ProcessLookupError(process.get('errno', 0), 'Owned PID is unavailable')
        start = process['proc_start_abstime']
        if self.samples and start != self.samples[0]['process']['proc_start_abstime']:
            raise RuntimeError('PID identity changed; refusing to sample reused PID')
        self.samples.append(row)
        elapsed = time.monotonic_ns() - wall
        self._sample_cpu_ns += time.thread_time_ns() - cpu
        self._sample_wall_ns += elapsed
        self._max_sample_wall_ns = max(self._max_sample_wall_ns, elapsed)
        return row

    def start(self):
        if self._thread is not None or self._closed:
            raise RuntimeError('Sampler may only start once')
        if self.report_path.exists():
            raise FileExistsError(self.report_path)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the evidence destination without truncating another report.
        self._reservation = self.report_path.open('x')
        self.started_at = time.time(); self.started_monotonic = time.monotonic()
        try:
            self._read('before')
        except BaseException:
            self._reservation.close()
            self._closed = True
            raise
        self._thread = threading.Thread(target=self._run, name='r10-os-telemetry', daemon=True)
        self._thread.start()
        return self

    def _run(self):
        deadline = self.started_monotonic + self.max_duration_s
        tick = self.started_monotonic + self.interval_s
        while not self._stop.wait(max(0.0, min(tick, deadline) - time.monotonic())):
            if time.monotonic() >= deadline:
                self.end_reason = 'duration_limit'
                return
            try:
                self._read('sample')
            except Exception as exc:
                self.errors.append({'at': time.time(), 'type': type(exc).__name__, 'detail': str(exc)})
                self.end_reason = 'target_unavailable_or_sampling_error'
                return
            tick += self.interval_s
            # Skip missed ticks; do not burst samples after OS scheduling delays.
            if tick < time.monotonic():
                tick = time.monotonic() + self.interval_s

    def mark(self, label):
        if not isinstance(label, str) or not label or len(label) > 256:
            raise ValueError('Nonempty label <=256 characters required')
        if self._thread is None or self._closed:
            raise RuntimeError('Sampler is not active')
        with self._lock:
            if len(self.marks) >= 1000:
                raise RuntimeError('Telemetry mark bound exceeded')
            self.marks.append({'label': label, 'at': time.time(), 'monotonic_ns': time.monotonic_ns()})

    def stop(self, reason='controller_finished'):
        if self._closed:
            return getattr(self, 'result', None)
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError('Telemetry thread did not stop within five seconds')
        try:
            self._read('after')
        except Exception as exc:
            self.errors.append({'at': time.time(), 'type': type(exc).__name__, 'detail': str(exc), 'phase': 'after'})
        finished = time.time()
        elapsed = max(time.monotonic() - self.started_monotonic, 1e-9)
        provenance = self.provenance
        if provenance is None:
            provenance = {'python': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
                          'native_abi': {'path': str(ABI_REPORT), 'sha256': sha(ABI_REPORT)},
                          'native_source': {'path': str(NATIVE), 'sha256': sha(NATIVE)},
                          'native_library': {'path': str(LIBRARY), 'sha256': sha(LIBRARY)}}
        self.result = {'status': 'passed' if not self.errors else 'observation_incomplete',
                       'schema_version': 1, 'read_only': True, 'pid': self.pid, 'controller_pid': os.getpid(),
                       'started_at': self.started_at, 'finished_at': finished, 'elapsed_s': elapsed,
                       'requested_stop_reason': reason, 'end_reason': getattr(self, 'end_reason', reason),
                       'interval_s': self.interval_s, 'duration_limit_s': self.max_duration_s,
                       'sample_count': len(self.samples), 'samples': self.samples,
                       'before': self.samples[0], 'after': self.samples[-1] if self.samples[-1]['kind'] == 'after' else None,
                       'marks': self.marks, 'errors': self.errors, 'provenance': provenance,
                       'overhead': {'sample_cpu_s': self._sample_cpu_ns / 1e9,
                                    'sample_cpu_percent_one_core': 100 * self._sample_cpu_ns / 1e9 / elapsed,
                                    'sample_wall_s': self._sample_wall_ns / 1e9,
                                    'max_sample_wall_ms': self._max_sample_wall_ns / 1e6},
                       'units': {'process_memory_diskio': 'bytes', 'process_pageins': 'pages',
                                 'process_cpu_time': 'nanoseconds', 'proc_start_abstime': 'native absolute clock',
                                 'host_vm': 'pages except metadata', 'swap_usage': 'bytes',
                                 'sysctl': 'raw values; unavailable values explicitly marked'},
                       'limits': ['Host VM counters are system-wide and may be kernel-cached.',
                                  'Providers are sampled sequentially, not atomically.',
                                  'Footprint minus resident bytes is not a process compression counter.',
                                  'Host compressor counters do not identify the process responsible.',
                                  'Thermal sysctls may be unavailable; no state is inferred from absence.',
                                  'Between-sample peaks are unverified.'],
                       'GPU_calls': 0, 'HTTP_requests': 0, 'target_mutations': 0,
                       'subprocesses_per_sample': 0}
        try:
            json.dump(self.result, self._reservation, indent=2)
            self._reservation.write('\n')
            self._reservation.flush()
        finally:
            self._reservation.close()
            self._closed = True
        return self.result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--selftest', type=Path, help='Read-only 3s self-PID overhead observation, fresh JSON output')
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps({'status': prepare()['status'], 'abi_report': str(ABI_REPORT)}))
    elif args.selftest:
        sampler = Sampler(os.getpid(), args.selftest, max_duration_s=4).start()
        try:
            sampler.mark('read_only_self_pid_test')
            time.sleep(3)
        finally:
            result = sampler.stop('selftest_complete')
        assert result['status'] == 'passed' and result['sample_count'] >= 6
        assert result['overhead']['sample_cpu_percent_one_core'] < 1.0
        assert result['before']['host_vm']['available'] and result['after']['host_vm']['available']
        print(json.dumps({'status': 'passed', 'report': str(args.selftest), 'overhead': result['overhead']}))
    else:
        parser.error('Choose --prepare or --selftest; target sampling is controller-owned through Sampler')


if __name__ == '__main__':
    main()
