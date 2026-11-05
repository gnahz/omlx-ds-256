#!/usr/bin/env python3
"""R10 sampler with the public system thermal-state API, no GPU or HTTP.

The initial sysctl-only implementation and its evidence remain unchanged.
This adapter reuses its bounded sampler lifecycle with an SDK-typed native
backend that additionally calls NSProcessInfo.thermalState in-process.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE.parent))
from portable import state_dir
OUTPUT = state_dir() / "telemetry"
BASE_PATH = HERE / 'private_os_telemetry.py'
spec = importlib.util.spec_from_file_location('r10_os_telemetry_base', BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
NATIVE = HERE / 'private_os_telemetry_thermal.m'
LIBRARY = OUTPUT / 'private_os_telemetry_thermal.dylib'
ABI_REPORT = OUTPUT / 'private-os-telemetry-thermal-abi.json'
THERMAL_HEADER = 'System/Library/Frameworks/Foundation.framework/Headers/NSProcessInfo.h'


class NativeReader(base.NativeReader):
    def __init__(self, *, check_provenance=True):
        if check_provenance:
            proof = json.loads(ABI_REPORT.read_text())
            if proof['status'] != 'passed' or base.sha(NATIVE) != proof['source']['sha256'] or base.sha(LIBRARY) != proof['library']['sha256']:
                raise RuntimeError('Thermal telemetry source/library provenance changed')
        self.lib = ctypes.CDLL(str(LIBRARY), use_errno=True)
        self.lib.r10_abi_json.argtypes = []
        self.lib.r10_abi_json.restype = ctypes.c_char_p
        self.lib.r10_snapshot_json.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.r10_snapshot_json.restype = ctypes.c_int
        self.abi = json.loads(self.lib.r10_abi_json())
        self.buffer = ctypes.create_string_buffer(16384)


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    if LIBRARY.exists() or ABI_REPORT.exists():
        raise FileExistsError('Thermal backend already prepared; preserve its evidence')
    sdk = Path(subprocess.check_output(['xcrun', '--show-sdk-path'], text=True).strip())
    command = ['xcrun', 'clang', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
               '-dynamiclib', '-isysroot', str(sdk), str(NATIVE), '-framework', 'Foundation', '-o', str(LIBRARY)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    native = NativeReader(check_provenance=False)
    old = json.loads(base.ABI_REPORT.read_text())
    if native.abi != old['abi']:
        raise RuntimeError('Thermal backend changed the independently verified typed ABI')
    report = {'status': 'passed', 'cpu_only': True, 'created_at': time.time(),
              'source': {'path': str(NATIVE), 'sha256': base.sha(NATIVE)},
              'library': {'path': str(LIBRARY), 'sha256': base.sha(LIBRARY)},
              'compiler': command, 'sdk': str(sdk), 'platform': platform.platform(),
              'headers': [{'path': str(sdk / h), 'sha256': base.sha(sdk / h)} for h in (*base.HEADERS, THERMAL_HEADER)],
              'abi': native.abi, 'original_abi_evidence': {'path': str(base.ABI_REPORT), 'sha256': base.sha(base.ABI_REPORT)},
              'thermal_contract': {'provider': 'NSProcessInfo.thermalState',
                                   'scope': 'system thermal state observed by collector; not GPU clock or per-PID temperature',
                                   'enum': {'0': 'nominal', '1': 'fair', '2': 'serious', '3': 'critical'},
                                   'availability': 'macOS 10.10.3 and later, checked in native code',
                                   'raw_thermal_sysctl_unavailability_preserved': True}}
    base.exclusive_json(ABI_REPORT, report)
    return report


class Sampler(base.Sampler):
    def __init__(self, pid, report_path, *, interval_s=0.5, max_duration_s=1800,
                 reader=None, provenance=None):
        if provenance is None:
            paths = [Path(__file__).resolve(), BASE_PATH, NATIVE, LIBRARY, ABI_REPORT, base.ABI_REPORT]
            provenance = {'inputs': [{'path': str(p), 'sha256': base.sha(p)} for p in paths],
                          'thermal_scope': 'system thermal state observed by collector; not GPU clock or per-PID temperature'}
        super().__init__(pid, report_path, interval_s=interval_s, max_duration_s=max_duration_s,
                         reader=reader or NativeReader(), provenance=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--selftest', type=Path)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps({'status': prepare()['status'], 'abi_report': str(ABI_REPORT)}))
    elif args.selftest:
        sampler = Sampler(os.getpid(), args.selftest, max_duration_s=4).start()
        try:
            sampler.mark('read_only_self_pid_thermal_test')
            time.sleep(3)
        finally:
            result = sampler.stop('selftest_complete')
        assert result['status'] == 'passed' and result['sample_count'] >= 6
        assert result['overhead']['sample_cpu_percent_one_core'] < 1.0
        for row in result['samples']:
            assert row['host_vm']['available']
            thermal = row['system_thermal']
            assert thermal['available'] and thermal['raw'] in range(4)
            assert thermal['state'] == ('nominal', 'fair', 'serious', 'critical')[thermal['raw']]
            assert all(name in row['sysctl'] for name in ('kern.thermal_level', 'machdep.xcpm.cpu_thermal_level', 'machdep.xcpm.gpu_thermal_level'))
        print(json.dumps({'status': 'passed', 'report': str(args.selftest), 'thermal': result['before']['system_thermal'], 'overhead': result['overhead']}))
    else:
        parser.error('Choose --prepare or --selftest; private PID sampling is controller-owned')


if __name__ == '__main__':
    main()
