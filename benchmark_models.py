"""Benchmark script to measure FLOPs, MADDs, and parameters for all MobileCount variants."""

import torch
from config import cfg
from models.CC import CrowdCounter

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count

    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print("Warning: fvcore not installed. Install with: pip install fvcore")

try:
    from thop import profile, clever_format

    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("Warning: thop not installed. Install with: pip install thop")


def benchmark_model(model_name, input_size=(1, 3, 768, 1024)):
    """Benchmark a single model variant."""
    print(f"\n{'=' * 70}")
    print(f"Model: {model_name}")
    print(f"{'=' * 70}")

    try:
        # Create model
        model = CrowdCounter(cfg.GPU_ID, model_name)
        model.eval()

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Size (MB):            {total_params * 4 / 1024 / 1024:.2f}")

        # Create dummy input
        dummy_input = torch.randn(input_size).to("mps")

        # Use thop if available
        if HAS_THOP:
            try:
                model_copy = CrowdCounter(cfg.GPU_ID, model_name)
                model_copy.eval()
                macs, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
                macs_str, params_str = clever_format([macs, params], "%.3f")
                print(f"\n[thop]")
                print(f"  MACs:   {macs_str}")
                print(f"  Params: {params_str}")
            except Exception as e:
                print(f"\n[thop] Error: {e}")

        # Use fvcore if available
        if HAS_FVCORE:
            try:
                model_copy = CrowdCounter(cfg.GPU_ID, model_name)
                model_copy.eval()
                flops = FlopCountAnalysis(model_copy, dummy_input)
                total_flops = flops.total()

                print(f"\n[fvcore]")
                print(f"  FLOPs: {total_flops:,}")
                print(f"  GFLOPs: {total_flops / 1e9:.3f}")
            except Exception as e:
                print(f"\n[fvcore] Error: {e}")

        # Test forward pass
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\nOutput shape: {output.shape}")
        print(f"✓ Benchmark successful")

        return {"name": model_name, "params": total_params, "success": True}

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return {"name": model_name, "params": 0, "success": False}


def main():
    print("\n" + "=" * 70)
    print("MobileCount Variants - Computational Complexity Benchmark")
    print("=" * 70)
    print(f"Input size: (1, 3, 768, 1024)")

    if not HAS_FVCORE and not HAS_THOP:
        print("\n⚠️  WARNING: Neither fvcore nor thop is installed!")
        print("Install for detailed metrics:")
        print("  pip install fvcore")
        print("  pip install thop")

    models_to_test = [
        "MobileCount",  # Original V2-based
        "MobileCountV3Lite",  # Custom lightweight V3
        "MobileCountV3Small",  # Full V3 Small
        "MobileCountV3Large",  # Full V3 Large
    ]

    # Only test V4/V5 if timm is available
    try:
        import timm

        models_to_test.extend(
            [
                "MobileCountV4",  # V4 via timm
                "MobileCountV5",  # V5 via timm
            ]
        )
    except ImportError:
        print("\n⚠️  timm not installed - skipping V4/V5")
        print("Install with: pip install timm")

    results = []
    for model_name in models_to_test:
        result = benchmark_model(model_name)
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Parameters':>15} {'Status':>10}")
    print("-" * 70)

    for r in results:
        status = "✓" if r["success"] else "✗"
        params_str = f"{r['params']:,}" if r["success"] else "N/A"
        print(f"{r['name']:<25} {params_str:>15} {status:>10}")

    print("=" * 70)

    # Recommendations
    print("\n💡 Recommendations:")
    print("  - MobileCount: Original baseline, proven architecture")
    print("  - MobileCountV3Lite: Best balance of efficiency and modern features")
    print("  - MobileCountV3Small: More powerful, good for accuracy")
    print("  - MobileCountV3Large: Highest accuracy, most compute")
    print("  - MobileCountV4/V5: Latest architectures (requires timm)")


if __name__ == "__main__":
    main()
