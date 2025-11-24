#!/usr/bin/env python3
"""
Problem Parser for Geometry Benchmark
Extracts geometric requirements from Chinese problem text using LLM.
"""

import json
import re
import os
from typing import Dict, List, Any, Optional


class ProblemParser:
    """Parse Chinese geometry problems to extract required objects and conditions."""
    
    def __init__(self, llm_api_function=None):
        """
        Initialize parser with optional LLM API function.
        
        Args:
            llm_api_function: Function that takes a prompt and returns LLM response
        """
        self.llm_api = llm_api_function
    
    def parse_problem(self, problem_text: str, problem_id: str = None) -> Dict[str, Any]:
        """
        Parse a geometry problem and extract requirements.
        
        Args:
            problem_text: Chinese text describing the geometry problem
            problem_id: Optional problem identifier
            
        Returns:
            Dictionary with required_objects and verification_conditions
        """
        if self.llm_api:
            return self._parse_with_llm(problem_text, problem_id)
        else:
            # Fallback to rule-based parsing
            return self._parse_rule_based(problem_text, problem_id)
    
    def _parse_with_llm(self, problem_text: str, problem_id: str = None) -> Dict[str, Any]:
        """Use LLM API to parse problem text."""
        prompt = self._create_parsing_prompt(problem_text)
        
        try:
            response = self.llm_api(prompt)
            
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            parsed_data = json.loads(response)
            
            result = {
                "id": problem_id or "unknown",
                "subject": problem_text,
                "required_objects": parsed_data.get("required_objects", {}),
                "verification_conditions": parsed_data.get("verification_conditions", [])
            }
            return result
        except Exception as e:
            print(f"LLM parsing failed: {e}, falling back to rule-based parsing")
            return self._parse_rule_based(problem_text, problem_id)
    
    def _parse_rule_based(self, problem_text: str, problem_id: str = None) -> Dict[str, Any]:
        """
        Rule-based parsing as fallback or for manual annotation.
        Extracts basic patterns from Chinese text.
        """
        # Extract point names (uppercase letters)
        points = self._extract_points(problem_text)
        
        # Extract geometric relationships
        conditions = []
        
        # Check for parallel lines (平行 or ∥)
        if '平行' in problem_text or '∥' in problem_text:
            parallel_pairs = self._extract_parallel_lines(problem_text, points)
            for pair in parallel_pairs:
                conditions.append({
                    "type": "parallel",
                    "objects": pair
                })
        
        # Check for perpendicular (垂直 or ⊥)
        if '垂直' in problem_text or '⊥' in problem_text:
            perp_pairs = self._extract_perpendicular_lines(problem_text, points)
            for pair in perp_pairs:
                conditions.append({
                    "type": "perpendicular",
                    "objects": pair
                })
        
        # Check for angle conditions (角 and degree symbol °)
        angle_conditions = self._extract_angle_conditions(problem_text, points)
        conditions.extend(angle_conditions)
        
        # Check for bisector (平分)
        if '平分' in problem_text:
            bisector_conditions = self._extract_bisector_conditions(problem_text, points)
            conditions.extend(bisector_conditions)
        
        # Infer required objects from points
        required_objects = self._infer_objects_from_points(points, problem_text)
        
        return {
            "id": problem_id or "unknown",
            "subject": problem_text,
            "required_objects": required_objects,
            "verification_conditions": conditions
        }
    
    def _extract_points(self, text: str) -> List[str]:
        """Extract point names (typically uppercase letters in geometry)."""
        # Find all uppercase letters that appear to be point names
        points = re.findall(r'[A-Z]', text)
        # Remove duplicates while preserving order
        seen = set()
        unique_points = []
        for p in points:
            if p not in seen:
                seen.add(p)
                unique_points.append(p)
        return unique_points
    
    def _extract_parallel_lines(self, text: str, points: List[str]) -> List[List[List[str]]]:
        """Extract parallel line pairs from text."""
        parallel_pairs = []
        
        # Pattern: AB∥CD or AB平行CD
        pattern = r'([A-Z]{2})[∥平行]+([A-Z]{2})'
        matches = re.findall(pattern, text)
        
        for match in matches:
            line1 = list(match[0])  # Convert "AB" to ["A", "B"]
            line2 = list(match[1])  # Convert "CD" to ["C", "D"]
            parallel_pairs.append([line1, line2])
        
        return parallel_pairs
    
    def _extract_perpendicular_lines(self, text: str, points: List[str]) -> List[List[List[str]]]:
        """Extract perpendicular line pairs from text."""
        perp_pairs = []
        
        # Pattern: AB⊥CD or AB垂直CD
        pattern = r'([A-Z]{2})[⊥垂直]+([A-Z]{2})'
        matches = re.findall(pattern, text)
        
        for match in matches:
            line1 = list(match[0])
            line2 = list(match[1])
            perp_pairs.append([line1, line2])
        
        return perp_pairs
    
    def _extract_angle_conditions(self, text: str, points: List[str]) -> List[Dict[str, Any]]:
        """Extract angle equality or value conditions."""
        conditions = []
        
        # Pattern: ∠ABC=50° or ∠ABC=∠DEF
        # First, find angle with specific value
        angle_value_pattern = r'∠([A-Z]{3})[=等于]*(\d+)°'
        matches = re.findall(angle_value_pattern, text)
        
        for match in matches:
            angle_points = list(match[0])  # "ABC" -> ["A", "B", "C"]
            value = float(match[1])
            # IMPORTANT: points must be nested array [["A", "B", "C"]] for angle_value
            conditions.append({
                "type": "angle_value",
                "points": [angle_points],  # Nested: [["A", "B", "C"]]
                "value": value
            })
        
        # Pattern: single point angle like ∠A=80° (in triangle context)
        # This is ambiguous - try to infer the full 3-point angle from context
        single_angle_pattern = r'∠([A-Z])[=等于]*(\d+)°'
        matches = re.findall(single_angle_pattern, text)
        
        for match in matches:
            vertex = match[0]
            value = float(match[1])
            
            # Try to find triangle containing this vertex
            triangle_pattern = r'三角形([A-Z]{3})'
            triangle_matches = re.findall(triangle_pattern, text)
            
            for tri in triangle_matches:
                tri_points = list(tri)
                if vertex in tri_points:
                    # Create 3-point angle with vertex in the middle
                    idx = tri_points.index(vertex)
                    # Get the other two points
                    other_points = [p for p in tri_points if p != vertex]
                    if len(other_points) == 2:
                        angle_points = [other_points[0], vertex, other_points[1]]
                        conditions.append({
                            "type": "angle_value",
                            "points": [angle_points],  # Nested: [[P1, vertex, P2]]
                            "value": value
                        })
                        break
        
        # Pattern: angle equality ∠ABC=∠DEF
        angle_eq_pattern = r'∠([A-Z]{3})[=等于]+∠([A-Z]{3})'
        matches = re.findall(angle_eq_pattern, text)
        
        for match in matches:
            angle1 = list(match[0])
            angle2 = list(match[1])
            conditions.append({
                "type": "angle_equality",
                "points": [angle1, angle2]
            })
        
        return conditions
    
    def _extract_bisector_conditions(self, text: str, points: List[str]) -> List[Dict[str, Any]]:
        """Extract angle bisector conditions."""
        conditions = []
        
        # Pattern: XY平分∠ABC (XY bisects angle ABC)
        pattern = r'([A-Z]{2})平分∠([A-Z]{3})'
        matches = re.findall(pattern, text)
        
        for match in matches:
            bisector_line = list(match[0])
            angle_points = list(match[1])
            conditions.append({
                "type": "angle_bisector",
                "line": bisector_line,
                "angle_points": angle_points
            })
        
        return conditions
    
    def _infer_objects_from_points(self, points: List[str], text: str) -> Dict[str, List]:
        """Infer required geometric objects from points and text."""
        required_objects = {
            "points": points,
            "segments": [],
            "lines": [],
            "circles": [],
            "polygons": []
        }
        
        # Find segments mentioned in text (e.g., AB, BC, CD)
        segment_pattern = r'([A-Z]{2})'
        potential_segments = re.findall(segment_pattern, text)
        
        # Filter to valid segments (both points exist)
        for seg in potential_segments:
            if len(seg) == 2 and seg[0] in points and seg[1] in points:
                seg_list = [seg[0], seg[1]]
                if seg_list not in required_objects["segments"]:
                    required_objects["segments"].append(seg_list)
        
        # Check for triangles (三角形)
        if '三角形' in text:
            triangle_pattern = r'三角形([A-Z]{3})'
            matches = re.findall(triangle_pattern, text)
            for match in matches:
                if len(match) == 3:
                    triangle = list(match)
                    if triangle not in required_objects["polygons"]:
                        required_objects["polygons"].append(triangle)
        
        # Check for quadrilaterals (四边形)
        if '四边形' in text:
            quad_pattern = r'四边形([A-Z]{4})'
            matches = re.findall(quad_pattern, text)
            for match in matches:
                if len(match) == 4:
                    quad = list(match)
                    if quad not in required_objects["polygons"]:
                        required_objects["polygons"].append(quad)
        
        # Check for circles (圆 or ⊙)
        if '圆' in text or '⊙' in text:
            circle_pattern = r'[圆⊙]([A-Z])'
            matches = re.findall(circle_pattern, text)
            for match in matches:
                if match not in required_objects["circles"]:
                    required_objects["circles"].append({"center": match})
        
        return required_objects
    
    def _create_parsing_prompt(self, problem_text: str) -> str:
        """Create a prompt for LLM to parse geometry problem."""
        prompt = f"""Parse the following Chinese geometry problem and extract:
1. Required geometric objects (points, segments, lines, circles, polygons)
2. Geometric conditions that must be verified (parallel, perpendicular, angles, etc.)

Problem: {problem_text}

Return a JSON object with this structure:
{{
  "required_objects": {{
    "points": ["A", "B", "C", ...],
    "segments": [["A", "B"], ["B", "C"], ...],
    "lines": [["A", "B"], ["C", "D"], ...],
    "circles": [{{"center": "O", "radius_point": "A"}}, ...],
    "polygons": [["A", "B", "C"], ...]
  }},
  "verification_conditions": [
    {{"type": "parallel", "objects": [["A", "B"], ["C", "D"]]}},
    {{"type": "perpendicular", "objects": [["A", "B"], ["C", "D"]]}},
    {{"type": "angle_value", "points": [["A", "B", "C"]], "value": 50}},
    {{"type": "angle_equality", "points": [["A", "B", "C"], ["D", "E", "F"]]}},
    {{"type": "angle_bisector", "line": ["E", "G"], "angle_points": ["B", "E", "F"]}},
    ...
  ]
}}

IMPORTANT RULES:
1. For "angle_value" conditions, ALWAYS use 3 points in nested format [["P1", "P2", "P3"]] where P2 is the vertex of the angle.
2. If the problem text mentions "∠A=80°" in triangle ABC, convert it to [["B", "A", "C"]] or [["C", "A", "B"]] with A as the middle point (vertex).
3. NEVER use single-point angles like [["A"]] or flat arrays like ["A", "B", "C"] for angle_value.
4. The "points" field for angle_value must be a nested list: [["point1", "point2", "point3"]].

Only return the JSON object, no other text.
"""
        return prompt
    
    def parse_from_json(self, json_file: str) -> Dict[str, Any]:
        """
        Parse problem from JSON file (like GeoQA3 format).
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            Parsed problem data
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problem_text = data.get('subject', '')
        problem_id = str(data.get('id', 'unknown'))
        
        return self.parse_problem(problem_text, problem_id)
    
    def save_to_benchmark_format(self, parsed_data: Dict[str, Any], output_file: str):
        """Save parsed data to benchmark format JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)


def create_openai_api_function(model: str = "gpt-4o-mini", api_key: str = None):
    """
    Create an OpenAI API function for parsing.
    
    Args:
        model: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo")
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        
    Returns:
        Function that takes a prompt and returns LLM response
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(api_key=api_key)
    
    def api_function(prompt: str) -> str:
        """Call OpenAI API with the prompt."""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a geometry problem parser. You extract geometric objects and conditions from Chinese geometry problems and return them as JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    return api_function


# Example usage
if __name__ == "__main__":
    # Example: Parse the problem from 1.json
    
    # Option 1: Rule-based parsing (default)
    parser = ProblemParser()
    
    # Option 2: With OpenAI LLM (uncomment to use)
    # api_key = os.getenv("OPENAI_API_KEY")
    # if api_key:
    #     llm_function = create_openai_api_function(model="gpt-4o-mini", api_key=api_key)
    #     parser = ProblemParser(llm_api_function=llm_function)
    #     print("Using OpenAI API for parsing")
    # else:
    #     print("OPENAI_API_KEY not found, using rule-based parsing")
    
    # Test with the example problem text
    problem_text = "如图,AB∥CD,直线EF交AB于点E,交CD于点F,EG平分∠BEF,交CD于点G,∠EFG=50°,则∠EGF等于()"
    
    result = parser.parse_problem(problem_text, problem_id="1")
    
    print("Parsed Problem:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

