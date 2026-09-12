"""Test setup ordering without downloading or compiling ASTRA-Sim.

Git and compilation are stubbed; the setup entry point, checksums, and
Python binding imports run normally in an isolated temporary checkout.
"""

import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest


class SetupAstraTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        script = Path(__file__).with_name("setup_astra.sh").read_text()
        self.setup = self.root / "script/e4/backend/setup_astra.sh"
        self.write(self.setup, script)
        self.source = self.root / ".deps/astra-sim-table7"
        self.binding = self.source / "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py"
        self.binary = self.source / "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default"
        bundle = self.setup.parent / "bundle"
        driver = "scripts/measure_table7_backend_generality.py"
        self.write(bundle / "overlay" / driver, "pass\n")
        checksum = hashlib.sha256(b"pass\n").hexdigest()
        self.write(bundle / "CHECKSUMS.sha256", f"{checksum}  overlay/{driver}\n")
        self.write(bundle / "INSTALLED_CHECKSUMS.sha256", f"{checksum}  {driver}\n")
        self.write(bundle / "astra-ras-ns3.patch", "")
        self.write(bundle / "overlay/scripts/build_astra_ns3.sh", '''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p extern/network_backend/ns-3/build/scratch
cp /bin/true extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default
if [[ "${TEST_EMIT_BINDING:-1}" == 1 ]]; then
    printf 'SCHEMA_VERSION = 1\n' > extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py
fi
''')
        pins = dict(re.findall(r"^(upstream_commit|chakra_commit)=(\w+)$", script, re.M))
        commands = self.root / "bin"
        self.write(commands / "git", f'''#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == clone ]]; then
    mkdir -p "${{@: -1}}/extern/graph_frontend/chakra/schema/protobuf"
elif [[ "$*" == *"rev-parse HEAD" ]]; then
    if [[ "$2" == */chakra ]]; then
        echo {pins["chakra_commit"]}
    else
        echo {pins["upstream_commit"]}
    fi
fi
''')
        # Exclude installed Chakra packages and bytecode caches from the test.
        python = commands / "python"
        self.write(python, f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -B -S "$@"\n')
        for name in ("cmake", "protoc", "mpicxx"):
            self.write(commands / name, "#!/bin/sh\nexit 0\n")
        self.env = {
            **os.environ,
            "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
            "CANONICAL_PYTHON": str(python),
            "ASTRA_TABLE7_ROOT": str(self.source),
            "ASTRA_TABLE7_MIRROR": "github",
            "PYTHONPATH": "",
            "TEST_EMIT_BINDING": "1",
        }

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)

    def run_setup(self, action, expected=0):
        result = subprocess.run(
            ["bash", str(self.setup), action], cwd=self.root, env=self.env,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_fresh_build_and_check(self):
        self.assertFalse(self.source.exists())
        self.run_setup("build")
        self.assertTrue(self.binding.is_file())
        self.run_setup("check")
        self.binding.unlink()
        result = self.run_setup("check", expected=2)
        self.assertIn("Chakra protobuf Python bindings are unavailable", result.stderr)

    def test_resume_prepared_source_without_bindings(self):
        self.run_setup("prepare")
        self.assertTrue((self.source / ".flexmaya-table7-source").is_file())
        self.assertFalse(self.binding.exists())
        self.run_setup("build")
        self.run_setup("check")

    def test_build_rejects_missing_bindings(self):
        self.env["TEST_EMIT_BINDING"] = "0"
        result = self.run_setup("build", expected=2)
        self.assertTrue(self.binary.is_file(), "build must run before binding validation")
        self.assertIn("Chakra protobuf Python bindings are unavailable", result.stderr)


if __name__ == "__main__":
    unittest.main()
