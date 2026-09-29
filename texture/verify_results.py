#!/usr/bin/env python3
"""
Verification script to check oversampling results.
Run after the pipeline completes to verify outputs.
"""

import os
import glob
import argparse


def main():
    """Verify pipeline outputs."""
    parser = argparse.ArgumentParser(description="Verify oversampling pipeline outputs")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Directory containing synthetic_samples, visualizations, and logs."
    )
    args = parser.parse_args()

    print("="*80)
    print("OVERSAMPLING PIPELINE VERIFICATION")
    print("="*80)
    
    # Check output directories
    output_dir = os.path.join(args.base_dir, "synthetic_samples")
    viz_dir = os.path.join(args.base_dir, "visualizations")
    logs_dir = os.path.join(args.base_dir, "logs")

    print(f"\nBase directory: {os.path.abspath(args.base_dir)}")
    
    print("\n1. Checking output directories...")
    for dir_path in [output_dir, viz_dir, logs_dir]:
        exists = os.path.exists(dir_path)
        print(f"   {dir_path:<25} {'✓ EXISTS' if exists else '✗ MISSING'}")
    
    # Check visualizations
    print("\n2. Checking visualizations...")
    if os.path.exists(viz_dir):
        viz_files = glob.glob(os.path.join(viz_dir, "*.png"))
        print(f"   Found {len(viz_files)} visualization(s)")
        for f in sorted(viz_files):
            print(f"     - {os.path.basename(f)}")
    else:
        print("   ✗ Visualization directory not found")
    
    # Check logs
    print("\n3. Checking logs...")
    if os.path.exists(logs_dir):
        log_files = glob.glob(os.path.join(logs_dir, "*"))
        print(f"   Found {len(log_files)} log file(s)")
        for f in sorted(log_files):
            print(f"     - {os.path.basename(f)}")
    else:
        print("   ✗ Logs directory not found")
    
    # Check synthetic samples per class
    print("\n4. Checking synthetic samples per class...")
    if os.path.exists(output_dir):
        class_dirs = [d for d in os.listdir(output_dir) 
                      if os.path.isdir(os.path.join(output_dir, d))]
        class_dirs.sort()
        
        if len(class_dirs) == 0:
            print("   ✗ No class directories found")
        else:
            print(f"\n   {'Class':<40} {'Synthetic Samples':<20}")
            print("   " + "-"*60)
            
            total_synthetic = 0
            for class_name in class_dirs:
                class_path = os.path.join(output_dir, class_name)
                files = glob.glob(os.path.join(class_path, "*.txt"))
                n_files = len(files)
                total_synthetic += n_files
                print(f"   {class_name:<40} {n_files:<20}")
            
            print("   " + "-"*60)
            print(f"   {'TOTAL':<40} {total_synthetic:<20}")
    else:
        print("   ✗ Synthetic samples directory not found")
    
    # Check report
    print("\n5. Checking summary report...")
    report_path = os.path.join(logs_dir, 'oversampling_report.txt')
    if os.path.exists(report_path):
        print(f"   ✓ Report found: {report_path}")
        file_size = os.path.getsize(report_path)
        print(f"     Size: {file_size} bytes")
    else:
        print(f"   ✗ Report not found: {report_path}")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
