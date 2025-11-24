#!/usr/bin/env python3
"""
Convert GeoQA3 dataset to benchmark format.
"""

import os
import json
import argparse
from problem_parser import ProblemParser, create_openai_api_function
from benchmark_dataset import BenchmarkDataset, BenchmarkProblem, RequiredObjects, VerificationCondition
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def convert_geoqa_problem(json_file: str, parser: ProblemParser) -> BenchmarkProblem:
    """
    Convert a GeoQA JSON file to BenchmarkProblem format.
    
    Args:
        json_file: Path to GeoQA JSON file
        parser: ProblemParser instance
        
    Returns:
        BenchmarkProblem
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problem_id = str(data.get('id', 'unknown'))
    subject = data.get('subject', '')
    
    # Parse the problem
    parsed = parser.parse_problem(subject, problem_id)
    
    # Create BenchmarkProblem
    required_objects = RequiredObjects.from_dict(parsed['required_objects'])
    conditions = [
        VerificationCondition.from_dict(c) 
        for c in parsed['verification_conditions']
    ]
    
    metadata = {
        "source": "GeoQA3",
        "formal_point": data.get('formal_point', []),
        "choices": data.get('choices', []),
        "correct_label": data.get('label'),
        "comment": data.get('comment', '')
    }
    
    problem = BenchmarkProblem(
        id=problem_id,
        subject=subject,
        required_objects=required_objects,
        verification_conditions=conditions,
        metadata=metadata
    )
    
    return problem


def convert_geoqa_dataset(input_dir: str, output_file: str, limit: int = None, use_llm: bool = False, model: str = "gpt-4o-mini"):
    """
    Convert GeoQA3 dataset to benchmark format.
    
    Args:
        input_dir: Directory containing GeoQA JSON files
        output_file: Output benchmark dataset file
        limit: Maximum number of problems to convert (None for all)
        use_llm: Whether to use OpenAI LLM for parsing
        model: OpenAI model to use (default: gpt-4o-mini)
    """
    # Initialize parser with or without LLM
    if use_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
            print("Falling back to rule-based parsing")
            parser = ProblemParser()
        else:
            try:
                print(f"Using OpenAI API with model: {model}")
                llm_function = create_openai_api_function(model=model, api_key=api_key)
                parser = ProblemParser(llm_api_function=llm_function)
            except Exception as e:
                print(f"Error setting up OpenAI API: {e}")
                print("Falling back to rule-based parsing")
                parser = ProblemParser()
    else:
        parser = ProblemParser()
    
    dataset = BenchmarkDataset()
    
    # Get all JSON files
    json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
    
    if limit:
        json_files = json_files[:limit]
    
    print(f"Converting {len(json_files)} problems from GeoQA3...")
    
    for i, filename in enumerate(json_files):
        filepath = os.path.join(input_dir, filename)
        
        try:
            problem = convert_geoqa_problem(filepath, parser)
            dataset.add_problem(problem)
            
            if (i + 1) % 10 == 0:
                print(f"Converted {i + 1}/{len(json_files)} problems")
        
        except Exception as e:
            print(f"Error converting {filename}: {e}")
            continue
    
    # Save dataset
    dataset.save(output_file)
    print(f"\nConverted {len(dataset)} problems")
    print(f"Saved to: {output_file}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert GeoQA3 to benchmark format")
    
    parser.add_argument("--input-dir", type=str, 
                       default="/Users/ooongs/Github/pyggb/data-5/GeoQA3/json",
                       help="Input directory with GeoQA JSON files")
    
    parser.add_argument("--output", type=str,
                       default="benchmark_geoqa3.json",
                       help="Output benchmark file")
    
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of problems to convert")
    
    parser.add_argument("--sample", action="store_true",
                       help="Convert only first 10 problems as sample")
    
    parser.add_argument("--use-llm", action="store_true",
                       help="Use OpenAI LLM for parsing (requires OPENAI_API_KEY in .env)")
    
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="OpenAI model to use (default: gpt-4o-mini)")
    
    args = parser.parse_args()
    
    limit = 10 if args.sample else args.limit
    
    convert_geoqa_dataset(args.input_dir, args.output, limit=limit, 
                         use_llm=args.use_llm, model=args.model)


if __name__ == "__main__":
    main()

