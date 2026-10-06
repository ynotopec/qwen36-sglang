"""Exercise the real wrappers with Docker/SGLang doubles; no GPU required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LaunchProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.project = self.directory / "project"
        self.project.mkdir()
        for name in ("run.sh", "entrypoint.sh", ".env.example"):
            shutil.copy2(ROOT / name, self.project / name)
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        for command in ("docker", "sglang"):
            executable = self.bin / command
            executable.write_text(
                f"#!{shutil.which('python3')}\n"
                "import json, os, sys\n"
                "if sys.argv[1:2] != ['rm']:\n"
                "    with open(os.environ['CAPTURE'], 'w') as f:\n"
                "        json.dump(sys.argv[1:], f)\n"
            )
            executable.chmod(0o755)
        self.capture = self.directory / "args.json"
        self.environment = {
            "PATH": f"{self.bin}:{os.defpath}",
            "HOME": str(self.directory),
            "CAPTURE": str(self.capture),
        }

    def launch(self, profile=""):
        (self.project / ".env").write_text("API_KEY=wrapper-test-only\n" + profile)
        result = subprocess.run(
            ["bash", "run.sh"], cwd=self.project, env=self.environment,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        docker_args = json.loads(self.capture.read_text())
        container_environment = dict(self.environment)
        for index, value in enumerate(docker_args):
            if value == "-e":
                name, setting = docker_args[index + 1].split("=", 1)
                container_environment[name] = setting
        result = subprocess.run(
            ["bash", "entrypoint.sh"], cwd=self.project,
            env=container_environment, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return docker_args, json.loads(self.capture.read_text())

    def test_optional_profile_end_to_end(self):
        example = (ROOT / ".env.example").read_text()
        block = example.split("# Optional NVIDIA Qwen3.8 Flash Next NVFP4 / NEXTN profile.", 1)[1]
        profile = "\n".join(
            line[2:] for line in block.splitlines()
            if line.startswith("# ") and "=" in line
            and line[2:].split("=", 1)[0].replace("_", "").isalnum()
        )
        nvme = self.directory / "local nvme"
        profile += f"\nPLE_OFFLOAD_HOST_DIR='{nvme}'\n"
        docker, args = self.launch(profile)
        expected = {
            "--model-path": "nvidia/Qwen3.8-Flash-Next-NVFP4",
            "--tp-size": "1", "--moe-runner-backend": "flashinfer_cutlass",
            "--fp4-gemm-backend": "flashinfer_cutlass", "--page-size": "64",
            "--chunked-prefill-size": "4096", "--context-length": "262144",
            "--speculative-algo": "NEXTN", "--speculative-num-steps": "3",
            "--speculative-eagle-topk": "1", "--speculative-num-draft-tokens": "4",
            "--max-running-requests": "8", "--max-mamba-cache-size": "40",
            "--reasoning-parser": "qwen3", "--mem-fraction-static": "0.85",
            "--host": "0.0.0.0", "--port": "30000",
            "--ple-offload-backend": "file", "--ple-offload-dir": "/app/ple",
        }
        for flag, value in expected.items():
            self.assertEqual(args[args.index(flag) + 1], value, flag)
        self.assertIn("--ple-offload-embedding", args)
        for flag in ("--sleep-on-idle", "--tool-call-parser", "--mamba-radix-cache-strategy"):
            self.assertNotIn(flag, args)
        self.assertIn("0.0.0.0:30000:30000", docker)
        self.assertIn(f"type=bind,source={nvme},target=/app/ple", docker)
        self.assertIn("--rm", docker)

    def test_default_profile_does_not_enable_new_flags(self):
        docker, args = self.launch()
        self.assertEqual(args[args.index("--model-path") + 1], "nvidia/Qwen3.6-35B-A3B-NVFP4")
        for flag in ("--fp4-gemm-backend", "--page-size", "--max-mamba-cache-size",
                     "--ple-offload-embedding", "--ple-offload-backend", "--ple-offload-dir"):
            self.assertNotIn(flag, args)
        self.assertNotIn("--mount", docker)

    def test_disabled_ple_ignores_directory_and_backend(self):
        docker, args = self.launch(
            "PLE_OFFLOAD_EMBEDDING=0\nPLE_OFFLOAD_BACKEND=file\n"
            "PLE_OFFLOAD_DIR=/app/ple\nPLE_OFFLOAD_HOST_DIR=/unused\n"
            "SGLANG_CACHE_DIR=/app/models/sglang\n"
        )
        self.assertNotIn("--ple-offload-dir", args)
        self.assertNotIn("--ple-offload-backend", args)
        self.assertNotIn("--mount", docker)
        self.assertIn("SGLANG_CACHE_DIR=/app/models/sglang", docker)

    def test_invalid_mount_is_rejected_before_docker(self):
        (self.project / ".env").write_text(
            "API_KEY=wrapper-test-only\nPLE_OFFLOAD_EMBEDDING=1\n"
            "PLE_OFFLOAD_HOST_DIR=relative/path\n"
        )
        result = subprocess.run(
            ["bash", "run.sh"], cwd=self.project, env=self.environment,
            capture_output=True, text=True, timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be absolute", result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
