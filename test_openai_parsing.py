#!/usr/bin/env python3
"""
Test OpenAI LLM Parsing
"""

import os
import json
from dotenv import load_dotenv
from problem_parser import ProblemParser, create_openai_api_function

# Load environment variables
load_dotenv()

def test_parsing():
    """Test both rule-based and LLM parsing."""
    
    test_problems = [
        "如图,AB∥CD,直线EF交AB于点E,交CD于点F,EG平分∠BEF,交CD于点G,∠EFG=50°,则∠EGF等于()",
        "如图,点O在直线AB上且OC⊥OD,点C、D在AB的同侧,若∠COA=36°则∠DOB的大小为()",
        "如图,△ABC的周长为30cm,把△ABC的边AC对折,使顶点C和点A重合,折痕交BC边于点D,交AC边与点E,连接AD,若AE=4cm,则△ABD的周长是()"
    ]
    
    print("="*70)
    print("OpenAI LLM Parsing Test")
    print("="*70)
    print()
    
    # Check if API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print()
        print("To use OpenAI LLM parsing:")
        print("1. Get API key from: https://platform.openai.com/api-keys")
        print("2. Create .env file: cp .env.example .env")
        print("3. Add your key: OPENAI_API_KEY=sk-proj-xxxxx")
        print()
        print("For now, testing with rule-based parsing only...")
        print()
        use_llm = False
    else:
        print(f"✓ Found OPENAI_API_KEY: {api_key[:20]}...")
        print()
        
        # Ask user if they want to use LLM (costs money)
        print("Would you like to test OpenAI LLM parsing?")
        print("(This will cost ~$0.001 per problem)")
        response = input("Use LLM? (y/N): ").strip().lower()
        use_llm = (response == 'y')
        print()
    
    # Create parsers
    rule_parser = ProblemParser()
    
    if use_llm:
        try:
            print("Initializing OpenAI API...")
            llm_function = create_openai_api_function(model="gpt-4o-mini", api_key=api_key)
            llm_parser = ProblemParser(llm_api_function=llm_function)
            print("✓ OpenAI API initialized")
            print()
        except Exception as e:
            print(f"✗ Error initializing OpenAI API: {e}")
            print("Falling back to rule-based only")
            use_llm = False
            print()
    
    # Test each problem
    for i, problem_text in enumerate(test_problems, 1):
        print("-"*70)
        print(f"Problem {i}:")
        print(f"{problem_text[:80]}...")
        print()
        
        # Rule-based parsing
        print("📋 Rule-based Parsing:")
        try:
            result_rule = rule_parser.parse_problem(problem_text, problem_id=f"test_{i}")
            print(f"  Points: {result_rule['required_objects']['points']}")
            print(f"  Conditions: {len(result_rule['verification_conditions'])} found")
            for cond in result_rule['verification_conditions']:
                print(f"    - {cond['type']}")
        except Exception as e:
            print(f"  Error: {e}")
        print()
        
        # LLM parsing
        if use_llm:
            print("🤖 LLM Parsing (OpenAI):")
            try:
                result_llm = llm_parser.parse_problem(problem_text, problem_id=f"test_{i}_llm")
                print(f"  Points: {result_llm['required_objects']['points']}")
                print(f"  Conditions: {len(result_llm['verification_conditions'])} found")
                for cond in result_llm['verification_conditions']:
                    print(f"    - {cond['type']}")
                print()
                
                # Compare results
                rule_points = set(result_rule['required_objects']['points'])
                llm_points = set(result_llm['required_objects']['points'])
                
                if rule_points == llm_points:
                    print("  ✓ Points match between rule-based and LLM")
                else:
                    print(f"  ⚠ Points differ:")
                    print(f"    Rule only: {rule_points - llm_points}")
                    print(f"    LLM only: {llm_points - rule_points}")
            except Exception as e:
                print(f"  Error: {e}")
            print()
    
    print("="*70)
    print("Test Complete!")
    print("="*70)


if __name__ == "__main__":
    test_parsing()

