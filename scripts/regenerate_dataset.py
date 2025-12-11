#!/usr/bin/env python3
"""
Regenerate Benchmark Dataset
Regenerate the benchmark dataset with improved parser and validation.
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from scripts.convert_geoqa_to_benchmark import convert_geoqa_dataset
from src.benchmark.benchmark_dataset import BenchmarkDataset
from src.utils import get_data_dir, get_data_dir

# Load environment variables
load_dotenv()


def main():
    """Regenerate the benchmark dataset."""
    print("="*70)
    print("BENCHMARK DATASET REGENERATION")
    print("="*70)
    
    # Configuration - use portable paths
    input_dir = str(get_data_dir() / "GeoQA3" / "json")
    output_file = str(get_data_dir() / "benchmark_geoqa3_new.json")
    limit = 10  # Set to a number to test with fewer problems
    use_llm = True  # Use LLM for parsing
    model = "gpt-4.1-mini"
    
    print(f"\nInput Directory: {input_dir}")
    print(f"Output File: {output_file}")
    print(f"Use LLM: {use_llm}")
    if use_llm:
        print(f"Model: {model}")
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"✗ Error: Input directory does not exist: {input_dir}")
        print("Please create the data directory or update the input_dir path.")
        return 1
    
    # Check for API key if using LLM
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        print("✗ Warning: OPENAI_API_KEY not found in environment")
        print("Falling back to rule-based parsing...")
        use_llm = False
    
    try:
        # Convert dataset
        print("Converting GeoQA3 dataset...")
        dataset = convert_geoqa_dataset(
            input_dir=input_dir,
            output_file=output_file,
            limit=limit,
            use_llm=use_llm,
            model=model
        )
        
        print(f"\n✓ Dataset regenerated successfully!")
        print(f"Total problems: {len(dataset)}")
        
        # Validate compatibility
        print("\nValidating dataset compatibility...")
        validation_result = dataset.validate_compatibility(verbose=True)
        
        if validation_result["compatible"]:
            print("\n✓ Dataset is fully compatible with validator!")
            
            # Backup old dataset if exists
            old_file = "benchmark_geoqa3.json"
            if os.path.exists(old_file):
                backup_file = "benchmark_geoqa3_backup.json"
                print(f"\nBacking up old dataset to: {backup_file}")
                os.rename(old_file, backup_file)
            
            # Rename new dataset
            print(f"Renaming {output_file} to {old_file}")
            os.rename(output_file, old_file)
            
            print("\n✓ Dataset regeneration complete!")
            return 0
        else:
            print(f"\n⚠ Dataset has {validation_result['problems_with_issues']} problems with unsupported conditions")
            print(f"Saved to: {output_file}")
            print("Review the validation report above and fix issues if needed.")
            return 0
        
    except Exception as e:
        print(f"\n✗ Error during dataset regeneration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

