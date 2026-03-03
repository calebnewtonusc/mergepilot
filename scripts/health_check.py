"""
Health check for all MergePilot services.

Checks:
  - vLLM synthesis servers (ports 8001-8004)
  - MergePilot API server (port 8000)
  - GPU memory utilization
"""

import asyncio
import json
import sys
from typing import Optional

import aiohttp


async def check_vllm(session: aiohttp.ClientSession, port: int) -> dict:
    """Check if a vLLM server is healthy."""
    url = f"http://localhost:{port}/health"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                return {"port": port, "status": "healthy"}
            return {"port": port, "status": f"http_{resp.status}"}
    except Exception as e:
        return {"port": port, "status": "down", "error": str(e)[:50]}


async def check_api(session: aiohttp.ClientSession) -> dict:
    """Check if the MergePilot API is running."""
    url = "http://localhost:8000/health"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {"service": "api", "status": "healthy", "model": data.get("model", "unknown")}
            return {"service": "api", "status": f"http_{resp.status}"}
    except Exception as e:
        return {"service": "api", "status": "down", "error": str(e)[:50]}


def check_gpu_memory() -> list[dict]:
    """Check GPU memory usage."""
    try:
        import torch
        if not torch.cuda.is_available():
            return [{"error": "no GPU"}]

        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = props.total_memory / 1024**3
            gpus.append({
                "index": i,
                "name": props.name,
                "allocated_gb": round(allocated, 1),
                "total_gb": round(total, 1),
                "utilization_pct": round(allocated / total * 100, 1),
            })
        return gpus
    except ImportError:
        return [{"error": "torch not installed"}]


async def main():
    print("━━━ MergePilot Health Check ━━━\n")

    all_healthy = True

    async with aiohttp.ClientSession() as session:
        # Check vLLM servers
        print("vLLM Synthesis Servers:")
        vllm_tasks = [check_vllm(session, port) for port in range(8001, 8005)]
        vllm_results = await asyncio.gather(*vllm_tasks)
        for r in vllm_results:
            status_icon = "OK" if r["status"] == "healthy" else "!!"
            print(f"  [{status_icon}] Port {r['port']}: {r['status']}")
            if r["status"] != "healthy":
                all_healthy = False

        print()

        # Check MergePilot API
        print("MergePilot API:")
        api_result = await check_api(session)
        status_icon = "OK" if api_result["status"] == "healthy" else "!!"
        print(f"  [{status_icon}] Port 8000: {api_result['status']}")
        if "model" in api_result:
            print(f"       Model: {api_result['model']}")
        if api_result["status"] != "healthy":
            all_healthy = False

    print()

    # Check GPUs
    print("GPU Memory:")
    gpus = check_gpu_memory()
    for gpu in gpus:
        if "error" in gpu:
            print(f"  [!!] {gpu['error']}")
            continue
        util = gpu["utilization_pct"]
        status = "OK" if util < 90 else "!!"
        print(
            f"  [{status}] GPU {gpu['index']} ({gpu['name']}): "
            f"{gpu['allocated_gb']:.1f}/{gpu['total_gb']:.1f} GB ({util:.0f}%)"
        )

    print()
    if all_healthy:
        print("All services healthy.")
    else:
        print("Some services down. Check logs in ./logs/")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
