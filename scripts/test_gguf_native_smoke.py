#!/usr/bin/env python3
"""
GGUF Quantization Smoke Test for the InferFlux CUDA backend

Tests each supported quantization type by:
1. Loading pre-quantized GGUF models
2. Running inference with the InferFlux CUDA backend
3. Validating outputs are generated correctly
4. Comparing performance across quantization types

Usage: python3 scripts/test_gguf_native_smoke.py --model-dir /path/to/gguf/models
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ANSI colors
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

# Configuration
SUPPORTED_QUANTIZATIONS = [
    "q4_k_m",
    "q5_k_m", 
    "q6_k",
    "q8_0",
]

TEST_PROMPT = "Hello, how are you?"
NUM_TOKENS = 10
TEMPERATURE = 0.0


def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def log_success(msg: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def log_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def print_header(title: str):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


class GGUFModelTester:
    """Test harness for GGUF models with the InferFlux CUDA backend"""
    
    def __init__(self, model_dir: Path, inferfluxd: str, inferctl: str):
        self.model_dir = model_dir
        self.inferfluxd = Path(inferfluxd)
        self.inferctl = Path(inferctl)
        self.test_results = {}
        self.test_artifacts = Path("/tmp/gguf_native_smoke")
        self.test_artifacts.mkdir(exist_ok=True)
        
    def find_gguf_models(self) -> Dict[str, Path]:
        """Find all GGUF models in the model directory"""
        models = {}
        
        if not self.model_dir.exists():
            log_error(f"Model directory not found: {self.model_dir}")
            return models
            
        # Find all .gguf files
        for gguf_file in self.model_dir.glob("*.gguf"):
            filename = gguf_file.name.lower()
            
            # Determine quantization type from filename
            quant_type = None
            for qt in SUPPORTED_QUANTIZATIONS:
                if qt.replace("_", "") in filename:
                    quant_type = qt
                    break
            
            if quant_type:
                models[quant_type] = gguf_file
                log_info(f"Found {quant_type}: {gguf_file.name}")
        
        return models
    
    def create_test_config(self, model_path: Path, quant_type: str, port: int) -> Path:
        """Create test configuration file"""
        config_path = self.test_artifacts / f"config_{quant_type}_{port}.yaml"
        
        config = {
            "server": {
                "host": "127.0.0.1",
                "port": port
            },
            "models": [{
                "id": f"test-{quant_type}",
                "path": str(model_path.absolute()),
                "format": "gguf",
                "backend": "inferflux_cuda",
                "default": True
            }],
            "runtime": {
                "cuda": {
                    "enabled": True
                }
            },
            "logging": {
                "level": "warning"
            }
        }
        
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def start_server(self, config_path: Path, quant_type: str) -> Tuple[subprocess.Popen, int]:
        """Start inferfluxd server"""
        port = 18083 + SUPPORTED_QUANTIZATIONS.index(quant_type)
        
        log_info(f"Starting server for {quant_type} on port {port}...")
        
        env = os.environ.copy()
        env["INFERFLUX_PORT_OVERRIDE"] = str(port)
        env["INFERCTL_API_KEY"] = "dev-key-123"
        
        log_file = self.test_artifacts / f"server_{quant_type}.log"
        
        with open(log_file, 'w') as lf:
            proc = subprocess.Popen(
                [str(self.inferfluxd), "--config", str(config_path)],
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT
            )
        
        # Wait for server to be ready
        max_wait = 30
        waited = 0
        while waited < max_wait:
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/livez", timeout=1) as response:
                    if response.status in (200, 401):
                        log_success(f"Server ready (PID: {proc.pid})")
                        return proc, port
            except:
                pass
            
            time.sleep(1)
            waited += 1
        
        log_error("Server did not start in time")
        with open(log_file, 'r') as f:
            print(f.read()[-500:])
        proc.kill()
        return None, port
    
    def run_inference(self, port: int, quant_type: str) -> Tuple[bool, str, int]:
        """Run inference test"""
        output_file = self.test_artifacts / f"output_{quant_type}.txt"
        
        log_info(f"Running inference for {quant_type}...")
        
        env = os.environ.copy()
        env["INFERCTL_API_KEY"] = "dev-key-123"
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [
                    str(self.inferctl), "chat",
                    "--host", "127.0.0.1",
                    "--port", str(port),
                    "--model", f"test-{quant_type}",
                    "--message", TEST_PROMPT,
                    "--max-tokens", str(NUM_TOKENS),
                    "--temperature", str(TEMPERATURE)
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=60
            )
            
            end_time = time.time()
            duration_ms = int((end_time - start_time) * 1000)
            
            # Save output
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            
            if result.returncode != 0:
                log_error(f"Inference failed with code {result.returncode}")
                log_error(f"stderr: {result.stderr}")
                return False, "", duration_ms
            
            output_text = result.stdout.strip()
            log_info(f"Generated output in {duration_ms}ms: {output_text[:100]}...")
            
            return True, output_text, duration_ms
            
        except subprocess.TimeoutExpired:
            log_error("Inference timed out")
            return False, "", 0
        except Exception as e:
            log_error(f"Inference exception: {e}")
            return False, "", 0
    
    def test_quantization(self, quant_type: str, model_path: Path) -> bool:
        """Test a single quantization type"""
        print_header(f"Testing: {quant_type}")
        
        # Create config
        config_path = self.create_test_config(model_path, quant_type, 0)
        
        # Start server
        server_proc, port = self.start_server(config_path, quant_type)
        
        if server_proc is None:
            self.test_results[quant_type] = "SERVER_FAILED"
            return False
        
        try:
            # Run inference
            success, output, duration_ms = self.run_inference(port, quant_type)
            
            if not success:
                self.test_results[quant_type] = "INFERENCE_FAILED"
                return False
            
            if not output or len(output) < 5:
                log_error(f"Output too short or empty: '{output}'")
                self.test_results[quant_type] = "EMPTY_OUTPUT"
                return False
            
            self.test_results[quant_type] = {
                "status": "SUCCESS",
                "duration_ms": duration_ms,
                "output_length": len(output),
                "output_preview": output[:100]
            }
            
            log_success(f"✅ {quant_type}: SUCCESS ({duration_ms}ms)")
            return True
            
        finally:
            # Cleanup server
            if server_proc:
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                    server_proc.wait()
    
    def run_all_tests(self, models: Dict[str, Path]) -> bool:
        """Test all quantization types"""
        print_header("GGUF InferFlux CUDA Smoke Test")
        
        log_info(f"Model directory: {self.model_dir}")
        log_info(f"Test prompt: '{TEST_PROMPT}'")
        log_info(f"Tokens to generate: {NUM_TOKENS}")
        log_info(f"Found {len(models)} quantized models")
        
        if not models:
            log_error("No GGUF models found!")
            log_info("Expected files like: *q4_k_m*.gguf, *q5_k_m*.gguf, etc.")
            return False
        
        # Test each quantization
        for quant_type in SUPPORTED_QUANTIZATIONS:
            if quant_type not in models:
                log_warning(f"Skipping {quant_type} (model not found)")
                self.test_results[quant_type] = "NOT_FOUND"
                continue
            
            self.test_quantization(quant_type, models[quant_type])
            print()
        
        # Print results
        self.print_results()
        
        return self.get_success_count() > 0
    
    def get_success_count(self) -> int:
        """Count successful tests"""
        count = 0
        for result in self.test_results.values():
            if isinstance(result, dict) and result.get("status") == "SUCCESS":
                count += 1
        return count
    
    def print_results(self):
        """Print test results summary"""
        print_header("Test Results Summary")
        
        print()
        print(f"{'Quantization':<15} | {'Duration':<10} | {'Output Len':<10} | {'Status'}")
        print("-" * 60)
        
        for quant_type in SUPPORTED_QUANTIZATIONS:
            result = self.test_results.get(quant_type, "NOT_TESTED")
            
            if isinstance(result, dict):
                duration = result.get("duration_ms", "N/A")
                length = result.get("output_length", "N/A")
                status = result.get("status", "UNKNOWN")
                
                if status == "SUCCESS":
                    status = f"{Colors.GREEN}{status}{Colors.NC}"
                else:
                    status = f"{Colors.RED}{status}{Colors.NC}"
                
                print(f"{quant_type:<15} | {duration:<10} | {length:<10} | {status}")
            else:
                status = f"{Colors.YELLOW}{result}{Colors.NC}"
                print(f"{quant_type:<15} | {'N/A':<10} | {'N/A':<10} | {status}")
        
        print()
        
        total = len(SUPPORTED_QUANTIZATIONS)
        passed = self.get_success_count()
        
        log_info(f"Tests passed: {passed} / {total}")
        
        if passed == total:
            log_success("✅ All quantization types passed!")
        elif passed > 0:
            log_warning(f"⚠️  {passed}/{total} quantization types passed")
        else:
            log_error("❌ No quantization types passed")
        
        log_info(f"Test artifacts: {self.test_artifacts}")
        log_info("To cleanup: rm -rf " + str(self.test_artifacts))


def main():
    parser = argparse.ArgumentParser(
        description="GGUF InferFlux CUDA Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all quantizations in a directory
  python3 scripts/test_gguf_native_smoke.py --model-dir /path/to/gguf/models
  
  # Use custom binaries
  python3 scripts/test_gguf_native_smoke.py \\
      --model-dir /path/to/models \\
      --inferfluxd ./build/inferfluxd \\
      --inferctl ./build/inferctl

Environment Variables:
  MODEL_DIR       - Path to directory containing GGUF models
  INFERFLUXD_PATH - Path to inferfluxd binary
  INFERCTL_PATH   - Path to inferctl binary
        """
    )
    
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("MODEL_DIR", 
                                     os.path.expanduser("~/.inferflux/models"))),
        help="Directory containing GGUF models"
    )
    
    parser.add_argument(
        "--inferfluxd",
        default=os.environ.get("INFERFLUXD_PATH", "./build/inferfluxd"),
        help="Path to inferfluxd binary"
    )
    
    parser.add_argument(
        "--inferctl",
        default=os.environ.get("INFERCTL_PATH", "./build/inferctl"),
        help="Path to inferctl binary"
    )
    
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=NUM_TOKENS,
        help="Number of tokens to generate"
    )
    
    parser.add_argument(
        "--prompt",
        default=TEST_PROMPT,
        help="Test prompt"
    )
    
    args = parser.parse_args()
    
    # Update module-level defaults used by the test harness.
    globals()["NUM_TOKENS"] = args.num_tokens
    globals()["TEST_PROMPT"] = args.prompt
    
    # Validate binaries
    if not Path(args.inferfluxd).exists():
        log_error(f"inferfluxd not found: {args.inferfluxd}")
        log_info("Build it with: cmake --build build --target inferfluxd")
        return 1
    
    if not Path(args.inferctl).exists():
        log_error(f"inferctl not found: {args.inferctl}")
        log_info("Build it with: cmake --build build --target inferctl")
        return 1
    
    # Run tests
    tester = GGUFModelTester(args.model_dir, args.inferfluxd, args.inferctl)
    models = tester.find_gguf_models()
    
    success = tester.run_all_tests(models)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
