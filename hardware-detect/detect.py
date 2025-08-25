#!/usr/bin/env python3
"""
Unicorn Orator Hardware Detection System
Identifies available acceleration hardware and recommends optimal configuration
"""

import os
import subprocess
import json
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardwareDetector:
    """Comprehensive hardware detection for speech processing acceleration"""
    
    def __init__(self):
        self.system_info = {
            "os": platform.system(),
            "arch": platform.machine(),
            "processor": platform.processor(),
        }
        self.detected_hardware = {}
        
    def detect_all(self) -> Dict:
        """Run all hardware detection methods"""
        logger.info("🔍 Starting hardware detection...")
        
        self.detected_hardware = {
            "cpu": self.detect_cpu(),
            "nvidia_gpu": self.detect_nvidia_gpu(),
            "amd_npu": self.detect_amd_npu(),
            "intel_igpu": self.detect_intel_igpu(),
            "amd_igpu": self.detect_amd_igpu(),
        }
        
        return self.detected_hardware
    
    def detect_cpu(self) -> Dict:
        """Detect CPU capabilities"""
        cpu_info = {
            "available": True,
            "vendor": "unknown",
            "model": platform.processor(),
            "cores": os.cpu_count(),
            "features": []
        }
        
        try:
            # Check CPU features on Linux
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    if "avx2" in cpuinfo.lower():
                        cpu_info["features"].append("AVX2")
                    if "avx512" in cpuinfo.lower():
                        cpu_info["features"].append("AVX512")
                    if "AMD" in cpuinfo:
                        cpu_info["vendor"] = "AMD"
                    elif "Intel" in cpuinfo:
                        cpu_info["vendor"] = "Intel"
            
            logger.info(f"✅ CPU: {cpu_info['vendor']} with {cpu_info['cores']} cores")
        except Exception as e:
            logger.warning(f"CPU detection error: {e}")
            
        return cpu_info
    
    def detect_nvidia_gpu(self) -> Dict:
        """Detect NVIDIA GPU using nvidia-smi"""
        gpu_info = {
            "available": False,
            "devices": [],
            "cuda_version": None,
            "driver_version": None
        }
        
        try:
            # Check if nvidia-smi exists
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                gpu_info["available"] = True
                for line in result.stdout.strip().split("\n"):
                    name, memory = line.split(", ")
                    gpu_info["devices"].append({
                        "name": name,
                        "memory": memory
                    })
                
                # Get CUDA version
                result = subprocess.run(
                    ["nvidia-smi", "--query", "cuda_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    gpu_info["cuda_version"] = result.stdout.strip()
                
                logger.info(f"✅ NVIDIA GPU: {len(gpu_info['devices'])} device(s) found")
                for device in gpu_info["devices"]:
                    logger.info(f"  - {device['name']} ({device['memory']})")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("NVIDIA GPU not detected")
            
        return gpu_info
    
    def detect_amd_npu(self) -> Dict:
        """Detect AMD NPU (XDNA)"""
        npu_info = {
            "available": False,
            "type": None,
            "driver": None,
            "tops": None
        }
        
        try:
            # Check for AMD Ryzen AI
            # Method 1: Check for XDNA driver
            xdna_path = Path("/dev/xdna")
            if xdna_path.exists():
                npu_info["available"] = True
                npu_info["driver"] = "xdna"
            
            # Method 2: Check for NPU device nodes
            npu_devices = list(Path("/dev").glob("npu*"))
            if npu_devices:
                npu_info["available"] = True
                npu_info["driver"] = "npu"
            
            # Method 3: Check CPU model for known NPU-enabled processors
            if self.system_info["processor"]:
                processor = self.system_info["processor"].lower()
                # AMD Ryzen 7040 series (Phoenix) has 16 TOPS NPU
                if any(model in processor for model in ["7840", "7940", "7640", "7540"]):
                    npu_info["available"] = True
                    npu_info["type"] = "XDNA1"
                    npu_info["tops"] = 16
                # AMD Ryzen 8040 series (Hawk Point) has 16 TOPS NPU
                elif any(model in processor for model in ["8840", "8940", "8640", "8540"]):
                    npu_info["available"] = True
                    npu_info["type"] = "XDNA1"
                    npu_info["tops"] = 16
                # AMD Ryzen AI 300 series (Strix Point) has 50 TOPS NPU
                elif "ai 3" in processor:
                    npu_info["available"] = True
                    npu_info["type"] = "XDNA2"
                    npu_info["tops"] = 50
            
            # Method 4: Check for Ryzen AI Software
            ryzen_ai_path = Path("/opt/ryzen-ai-sw")
            if ryzen_ai_path.exists():
                npu_info["available"] = True
                npu_info["driver"] = "ryzen-ai-sw"
            
            if npu_info["available"]:
                logger.info(f"✅ AMD NPU: {npu_info['type']} detected ({npu_info['tops']} TOPS)")
            else:
                logger.debug("AMD NPU not detected")
                
        except Exception as e:
            logger.warning(f"AMD NPU detection error: {e}")
            
        return npu_info
    
    def detect_intel_igpu(self) -> Dict:
        """Detect Intel integrated GPU"""
        igpu_info = {
            "available": False,
            "device": None,
            "driver": None,
            "compute_units": None
        }
        
        try:
            # Check for Intel GPU device
            render_devices = list(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").exists() else []
            
            for device in render_devices:
                # Try to identify if it's Intel
                try:
                    # Method 1: Check with vainfo
                    result = subprocess.run(
                        ["vainfo", f"--display", "drm", f"--device", str(device)],
                        capture_output=True, text=True, timeout=5
                    )
                    if "Intel" in result.stdout:
                        igpu_info["available"] = True
                        igpu_info["device"] = str(device)
                        igpu_info["driver"] = "i915" if "i915" in result.stdout else "xe"
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
                
                # Method 2: Check sysfs
                try:
                    vendor_path = Path(f"/sys/class/drm/card{device.name[-1]}/device/vendor")
                    if vendor_path.exists():
                        with open(vendor_path) as f:
                            vendor = f.read().strip()
                            if vendor == "0x8086":  # Intel vendor ID
                                igpu_info["available"] = True
                                igpu_info["device"] = str(device)
                                break
                except Exception:
                    pass
            
            # Method 3: Check for Intel GPU tools
            try:
                result = subprocess.run(
                    ["intel_gpu_frequency", "--get"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    igpu_info["available"] = True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            if igpu_info["available"]:
                logger.info(f"✅ Intel iGPU: Detected at {igpu_info['device']}")
            else:
                logger.debug("Intel iGPU not detected")
                
        except Exception as e:
            logger.warning(f"Intel iGPU detection error: {e}")
            
        return igpu_info
    
    def detect_amd_igpu(self) -> Dict:
        """Detect AMD integrated GPU"""
        igpu_info = {
            "available": False,
            "device": None,
            "driver": None,
            "model": None
        }
        
        try:
            # Check for AMD GPU device
            render_devices = list(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").exists() else []
            
            for device in render_devices:
                try:
                    # Check sysfs for AMD vendor
                    vendor_path = Path(f"/sys/class/drm/card{device.name[-1]}/device/vendor")
                    if vendor_path.exists():
                        with open(vendor_path) as f:
                            vendor = f.read().strip()
                            if vendor == "0x1002":  # AMD vendor ID
                                igpu_info["available"] = True
                                igpu_info["device"] = str(device)
                                igpu_info["driver"] = "amdgpu"
                                
                                # Try to get model info
                                model_path = Path(f"/sys/class/drm/card{device.name[-1]}/device/product")
                                if model_path.exists():
                                    with open(model_path) as f:
                                        igpu_info["model"] = f.read().strip()
                                break
                except Exception:
                    pass
            
            # Check if it's likely an iGPU based on processor
            if igpu_info["available"] and self.detected_hardware.get("cpu", {}).get("vendor") == "AMD":
                processor = self.system_info["processor"].lower()
                if any(term in processor for term in ["780m", "680m", "vega", "radeon"]):
                    igpu_info["model"] = "AMD Radeon Graphics (iGPU)"
            
            if igpu_info["available"]:
                logger.info(f"✅ AMD iGPU: {igpu_info['model'] or 'Detected'} at {igpu_info['device']}")
            else:
                logger.debug("AMD iGPU not detected")
                
        except Exception as e:
            logger.warning(f"AMD iGPU detection error: {e}")
            
        return igpu_info
    
    def recommend_configuration(self) -> Dict:
        """Recommend optimal configuration based on detected hardware"""
        recommendations = {
            "whisperx": {
                "backend": "cpu",
                "variant": "full",
                "reason": "Default CPU fallback"
            },
            "kokoro": {
                "backend": "cpu",
                "reason": "Default CPU fallback"
            }
        }
        
        # Priority order for WhisperX
        if self.detected_hardware.get("amd_npu", {}).get("available"):
            recommendations["whisperx"] = {
                "backend": "npu",
                "variant": "full",
                "reason": f"AMD NPU detected ({self.detected_hardware['amd_npu']['tops']} TOPS)"
            }
            recommendations["kokoro"] = {
                "backend": "npu",
                "reason": "AMD NPU provides excellent TTS acceleration"
            }
        elif self.detected_hardware.get("intel_igpu", {}).get("available"):
            recommendations["whisperx"] = {
                "backend": "igpu",
                "variant": "full",
                "reason": "Intel iGPU provides good acceleration via OpenVINO"
            }
            recommendations["kokoro"] = {
                "backend": "igpu",
                "reason": "Intel iGPU optimized with OpenVINO"
            }
        elif self.detected_hardware.get("nvidia_gpu", {}).get("available"):
            recommendations["whisperx"] = {
                "backend": "cuda",
                "variant": "full",
                "reason": f"NVIDIA GPU: {self.detected_hardware['nvidia_gpu']['devices'][0]['name']}"
            }
            recommendations["kokoro"] = {
                "backend": "cuda",
                "reason": "NVIDIA GPU provides fastest inference"
            }
        elif self.detected_hardware.get("amd_igpu", {}).get("available"):
            # AMD iGPU can use CPU optimized versions for now
            # Future: Add ROCm support
            recommendations["whisperx"]["reason"] = "AMD iGPU detected, using optimized CPU backend"
            recommendations["kokoro"]["reason"] = "AMD iGPU detected, using optimized CPU backend"
        
        return recommendations
    
    def generate_report(self) -> str:
        """Generate a human-readable hardware report"""
        report = ["🦄 Unicorn Orator Hardware Detection Report", "=" * 50, ""]
        
        # System Info
        report.append("System Information:")
        report.append(f"  OS: {self.system_info['os']}")
        report.append(f"  Architecture: {self.system_info['arch']}")
        report.append(f"  Processor: {self.system_info['processor']}")
        report.append("")
        
        # Detected Hardware
        report.append("Detected Hardware:")
        
        # CPU
        cpu = self.detected_hardware.get("cpu", {})
        if cpu.get("available"):
            report.append(f"  ✅ CPU: {cpu['vendor']} ({cpu['cores']} cores)")
            if cpu.get("features"):
                report.append(f"     Features: {', '.join(cpu['features'])}")
        
        # NVIDIA GPU
        nvidia = self.detected_hardware.get("nvidia_gpu", {})
        if nvidia.get("available"):
            report.append(f"  ✅ NVIDIA GPU: {len(nvidia['devices'])} device(s)")
            for device in nvidia["devices"]:
                report.append(f"     - {device['name']} ({device['memory']})")
        else:
            report.append("  ❌ NVIDIA GPU: Not detected")
        
        # AMD NPU
        amd_npu = self.detected_hardware.get("amd_npu", {})
        if amd_npu.get("available"):
            report.append(f"  ✅ AMD NPU: {amd_npu['type']} ({amd_npu['tops']} TOPS)")
        else:
            report.append("  ❌ AMD NPU: Not detected")
        
        # Intel iGPU
        intel_igpu = self.detected_hardware.get("intel_igpu", {})
        if intel_igpu.get("available"):
            report.append(f"  ✅ Intel iGPU: {intel_igpu['device']}")
        else:
            report.append("  ❌ Intel iGPU: Not detected")
        
        # AMD iGPU
        amd_igpu = self.detected_hardware.get("amd_igpu", {})
        if amd_igpu.get("available"):
            report.append(f"  ✅ AMD iGPU: {amd_igpu.get('model', 'Detected')}")
        else:
            report.append("  ❌ AMD iGPU: Not detected")
        
        report.append("")
        
        # Recommendations
        recommendations = self.recommend_configuration()
        report.append("Recommended Configuration:")
        report.append(f"  WhisperX: {recommendations['whisperx']['backend']} "
                     f"({recommendations['whisperx']['reason']})")
        report.append(f"  Kokoro: {recommendations['kokoro']['backend']} "
                     f"({recommendations['kokoro']['reason']})")
        
        return "\n".join(report)


def main():
    """Run hardware detection and print report"""
    detector = HardwareDetector()
    detector.detect_all()
    
    print(detector.generate_report())
    print("\n" + "=" * 50)
    
    # Save results to JSON
    results = {
        "hardware": detector.detected_hardware,
        "recommendations": detector.recommend_configuration(),
        "system": detector.system_info
    }
    
    output_file = Path("hardware_detection.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    main()