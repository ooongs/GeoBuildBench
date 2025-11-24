#!/usr/bin/env python3
"""
DSL Validator for Geometry Benchmark
Validates DSL files against required objects and verification conditions.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from random_constr import Construction
from benchmark_dataset import BenchmarkProblem, VerificationCondition
import geo_types as gt
import commands as cmd


@dataclass
class ValidationResult:
    """Result of validating a DSL against a benchmark problem."""
    success: bool
    object_score: float  # 0.0 to 1.0
    condition_score: float  # 0.0 to 1.0
    total_score: float  # 0.0 to 1.0
    missing_objects: Dict[str, List]
    failed_conditions: List[Dict[str, Any]]
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "object_score": self.object_score,
            "condition_score": self.condition_score,
            "total_score": self.total_score,
            "missing_objects": self.missing_objects,
            "failed_conditions": self.failed_conditions,
            "error_message": self.error_message,
            "details": self.details
        }


class DSLValidator:
    """Validate DSL constructions against benchmark requirements."""
    
    def __init__(self, tolerance: float = 1e-2):
        """
        Initialize validator.
        
        Args:
            tolerance: Numerical tolerance for geometric comparisons
        """
        self.tolerance = tolerance
        self.construction = Construction()
    
    def validate(self, dsl_file: str, problem: BenchmarkProblem, 
                 max_attempts: int = 100) -> ValidationResult:
        """
        Validate a DSL file against a benchmark problem.
        
        Args:
            dsl_file: Path to DSL file to validate
            problem: BenchmarkProblem to validate against
            max_attempts: Maximum attempts to generate valid construction
            
        Returns:
            ValidationResult with scores and details
        """
        try:
            # Load and generate construction
            self.construction.load(dsl_file)
            self.construction.generate(require_theorem=False, max_attempts=max_attempts)
            
            # Check required objects
            object_result = self._check_required_objects(problem.required_objects)
            
            # Check verification conditions
            condition_result = self._check_verification_conditions(
                problem.verification_conditions
            )
            
            # Calculate total score (weighted average)
            object_score = object_result["score"]
            condition_score = condition_result["score"]
            total_score = 0.3 * object_score + 0.7 * condition_score
            
            success = (object_score >= 0.9 and condition_score >= 0.9)
            
            return ValidationResult(
                success=success,
                object_score=object_score,
                condition_score=condition_score,
                total_score=total_score,
                missing_objects=object_result["missing"],
                failed_conditions=condition_result["failed"],
                details={
                    "object_details": object_result["details"],
                    "condition_details": condition_result["details"]
                }
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                object_score=0.0,
                condition_score=0.0,
                total_score=0.0,
                missing_objects={},
                failed_conditions=[],
                error_message=str(e)
            )
    
    def _check_required_objects(self, required_objects) -> Dict[str, Any]:
        """Check if all required objects exist in the construction."""
        element_dict = self.construction.element_dict
        
        missing = {
            "points": [],
            "segments": [],
            "lines": [],
            "circles": [],
            "polygons": []
        }
        
        found = {
            "points": [],
            "segments": [],
            "lines": [],
            "circles": [],
            "polygons": []
        }
        
        # Check points
        for point_label in required_objects.points:
            if point_label in element_dict:
                element = element_dict[point_label]
                if isinstance(element.data, gt.Point):
                    found["points"].append(point_label)
                else:
                    missing["points"].append(point_label)
            else:
                missing["points"].append(point_label)
        
        # Check segments
        for seg in required_objects.segments:
            seg_label = self._find_segment(seg[0], seg[1])
            if seg_label:
                found["segments"].append(seg)
            else:
                missing["segments"].append(seg)
        
        # Check lines
        for line in required_objects.lines:
            line_label = self._find_line(line[0], line[1])
            if line_label:
                found["lines"].append(line)
            else:
                missing["lines"].append(line)
        
        # Check circles
        for circle_def in required_objects.circles:
            center = circle_def.get("center")
            if center and center in element_dict:
                # Check if there's a circle with this center
                circle_label = self._find_circle_with_center(center)
                if circle_label:
                    found["circles"].append(circle_def)
                else:
                    missing["circles"].append(circle_def)
            else:
                missing["circles"].append(circle_def)
        
        # Check polygons
        for poly_points in required_objects.polygons:
            poly_label = self._find_polygon(poly_points)
            if poly_label:
                found["polygons"].append(poly_points)
            else:
                missing["polygons"].append(poly_points)
        
        # Calculate score
        total_required = (
            len(required_objects.points) +
            len(required_objects.segments) +
            len(required_objects.lines) +
            len(required_objects.circles) +
            len(required_objects.polygons)
        )
        
        total_found = (
            len(found["points"]) +
            len(found["segments"]) +
            len(found["lines"]) +
            len(found["circles"]) +
            len(found["polygons"])
        )
        
        score = total_found / total_required if total_required > 0 else 1.0
        
        return {
            "score": score,
            "missing": missing,
            "found": found,
            "details": {
                "total_required": total_required,
                "total_found": total_found
            }
        }
    
    def _find_segment(self, p1: str, p2: str) -> Optional[str]:
        """Find a segment between two points in the construction."""
        element_dict = self.construction.element_dict
        
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Segment):
                seg = element.data
                # Check if segment connects p1 and p2
                if p1 in element_dict and p2 in element_dict:
                    pt1 = element_dict[p1].data
                    pt2 = element_dict[p2].data
                    if isinstance(pt1, gt.Point) and isinstance(pt2, gt.Point):
                        # Check if segment endpoints match
                        if (np.allclose(seg.end_points[0], pt1.a, atol=self.tolerance) and
                            np.allclose(seg.end_points[1], pt2.a, atol=self.tolerance)):
                            return label
                        if (np.allclose(seg.end_points[0], pt2.a, atol=self.tolerance) and
                            np.allclose(seg.end_points[1], pt1.a, atol=self.tolerance)):
                            return label
        return None
    
    def _find_line(self, p1: str, p2: str) -> Optional[str]:
        """Find a line through two points in the construction."""
        element_dict = self.construction.element_dict
        
        if p1 not in element_dict or p2 not in element_dict:
            return None
        
        pt1 = element_dict[p1].data
        pt2 = element_dict[p2].data
        
        if not isinstance(pt1, gt.Point) or not isinstance(pt2, gt.Point):
            return None
        
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Line):
                line = element.data
                # Check if both points lie on the line
                if line.contains(pt1.a) and line.contains(pt2.a):
                    return label
        
        return None
    
    def _find_circle_with_center(self, center: str) -> Optional[str]:
        """Find a circle with given center point."""
        element_dict = self.construction.element_dict
        
        if center not in element_dict:
            return None
        
        center_point = element_dict[center].data
        if not isinstance(center_point, gt.Point):
            return None
        
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Circle):
                circle = element.data
                if np.allclose(circle.c, center_point.a, atol=self.tolerance):
                    return label
        
        return None
    
    def _find_polygon(self, points: List[str]) -> Optional[str]:
        """Find a polygon with given vertices."""
        element_dict = self.construction.element_dict
        
        # Check all points exist
        for p in points:
            if p not in element_dict or not isinstance(element_dict[p].data, gt.Point):
                return None
        
        # Get point coordinates
        point_coords = [element_dict[p].data.a for p in points]
        
        for label, element in element_dict.items():
            if isinstance(element.data, gt.Polygon):
                poly = element.data
                # Check if polygon has same vertices
                if len(poly.points) == len(point_coords):
                    # Check if all points match (order might differ)
                    if self._points_match_polygon(point_coords, poly.points):
                        return label
        
        return None
    
    def _points_match_polygon(self, points1: List[np.ndarray], 
                             points2: np.ndarray) -> bool:
        """Check if two sets of points match (considering rotation)."""
        n = len(points1)
        if n != len(points2):
            return False
        
        # Try all rotations
        for offset in range(n):
            match = True
            for i in range(n):
                if not np.allclose(points1[i], points2[(i + offset) % n], 
                                  atol=self.tolerance):
                    match = False
                    break
            if match:
                return True
        
        # Try reverse order
        for offset in range(n):
            match = True
            for i in range(n):
                if not np.allclose(points1[i], points2[(offset - i) % n], 
                                  atol=self.tolerance):
                    match = False
                    break
            if match:
                return True
        
        return False
    
    def _check_verification_conditions(self, conditions: List[VerificationCondition]) -> Dict[str, Any]:
        """Check all verification conditions."""
        failed = []
        passed = []
        details = []
        
        for condition in conditions:
            result = self._check_condition(condition)
            detail = {
                "condition": condition.to_dict(),
                "passed": result["passed"],
                "message": result["message"]
            }
            details.append(detail)
            
            if result["passed"]:
                passed.append(condition.to_dict())
            else:
                failed.append(condition.to_dict())
        
        score = len(passed) / len(conditions) if len(conditions) > 0 else 1.0
        
        return {
            "score": score,
            "failed": failed,
            "passed": passed,
            "details": details
        }
    
    def _check_condition(self, condition: VerificationCondition) -> Dict[str, Any]:
        """Check a single verification condition."""
        try:
            condition_type = condition.type
            
            if condition_type == "parallel":
                return self._check_parallel(condition.data)
            elif condition_type == "perpendicular":
                return self._check_perpendicular(condition.data)
            elif condition_type == "angle_value":
                return self._check_angle_value(condition.data)
            elif condition_type == "angle_equality":
                return self._check_angle_equality(condition.data)
            elif condition_type == "segment_equality":
                return self._check_segment_equality(condition.data)
            elif condition_type == "collinear":
                return self._check_collinear(condition.data)
            elif condition_type == "not_collinear":
                return self._check_not_collinear(condition.data)
            elif condition_type == "concyclic":
                return self._check_concyclic(condition.data)
            elif condition_type == "concurrent":
                return self._check_concurrent(condition.data)
            elif condition_type == "point_on_line":
                return self._check_point_on_line(condition.data)
            elif condition_type == "point_on_circle":
                return self._check_point_on_circle(condition.data)
            elif condition_type == "angle_bisector":
                return self._check_angle_bisector(condition.data)
            else:
                return {
                    "passed": False,
                    "message": f"Unknown condition type: {condition_type}"
                }
        except Exception as e:
            return {
                "passed": False,
                "message": f"Error checking condition: {str(e)}"
            }
    
    def _get_line_from_points(self, p1: str, p2: str) -> Optional[gt.Line]:
        """Get or create a line from two points."""
        element_dict = self.construction.element_dict
        
        if p1 not in element_dict or p2 not in element_dict:
            return None
        
        pt1 = element_dict[p1].data
        pt2 = element_dict[p2].data
        
        if not isinstance(pt1, gt.Point) or not isinstance(pt2, gt.Point):
            return None
        
        # Check if line already exists
        line_label = self._find_line(p1, p2)
        if line_label:
            return element_dict[line_label].data
        
        # Create line from points
        return cmd.line_pp(pt1, pt2)
    
    def _check_parallel(self, data: Dict) -> Dict[str, Any]:
        """Check if two lines are parallel."""
        objects = data.get("objects", [])
        if len(objects) != 2:
            return {"passed": False, "message": "Parallel condition requires 2 lines"}
        
        line1 = self._get_line_from_points(objects[0][0], objects[0][1])
        line2 = self._get_line_from_points(objects[1][0], objects[1][1])
        
        if line1 is None or line2 is None:
            return {"passed": False, "message": "Could not find lines"}
        
        result = cmd.are_parallel_ll(line1, line2)
        
        return {
            "passed": result.b,
            "message": f"Lines {'are' if result.b else 'are not'} parallel"
        }
    
    def _check_perpendicular(self, data: Dict) -> Dict[str, Any]:
        """Check if two lines are perpendicular."""
        objects = data.get("objects", [])
        if len(objects) != 2:
            return {"passed": False, "message": "Perpendicular condition requires 2 lines"}
        
        line1 = self._get_line_from_points(objects[0][0], objects[0][1])
        line2 = self._get_line_from_points(objects[1][0], objects[1][1])
        
        if line1 is None or line2 is None:
            return {"passed": False, "message": "Could not find lines"}
        
        result = cmd.are_perpendicular_ll(line1, line2)
        
        return {
            "passed": result.b,
            "message": f"Lines {'are' if result.b else 'are not'} perpendicular"
        }
    
    def _check_angle_value(self, data: Dict) -> Dict[str, Any]:
        """Check if angle has expected value."""
        points_list = data.get("points", [])
        expected_value = data.get("value", 0)
        tolerance = data.get("tolerance", 1.0)  # Default 1 degree tolerance
        
        if len(points_list) != 1 or len(points_list[0]) != 3:
            return {"passed": False, "message": "Angle value requires 3 points"}
        
        points = points_list[0]
        element_dict = self.construction.element_dict
        
        # Get points
        if not all(p in element_dict for p in points):
            return {"passed": False, "message": "Could not find all points"}
        
        p1 = element_dict[points[0]].data
        p2 = element_dict[points[1]].data
        p3 = element_dict[points[2]].data
        
        if not all(isinstance(p, gt.Point) for p in [p1, p2, p3]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Calculate angle
        angle = cmd.angle_ppp(p1, p2, p3)
        angle_degrees = np.degrees(angle.angle)
        
        # Check if angle matches expected value
        passed = np.abs(angle_degrees - expected_value) <= tolerance
        
        return {
            "passed": passed,
            "message": f"Angle is {angle_degrees:.2f}°, expected {expected_value}° (tolerance {tolerance}°)"
        }
    
    def _check_angle_equality(self, data: Dict) -> Dict[str, Any]:
        """Check if two angles are equal."""
        points_list = data.get("points", [])
        tolerance = data.get("tolerance", 1.0)
        
        if len(points_list) != 2:
            return {"passed": False, "message": "Angle equality requires 2 angles"}
        
        element_dict = self.construction.element_dict
        
        # Get first angle
        points1 = points_list[0]
        if len(points1) != 3 or not all(p in element_dict for p in points1):
            return {"passed": False, "message": "Could not find first angle points"}
        
        p1a, p1b, p1c = [element_dict[p].data for p in points1]
        if not all(isinstance(p, gt.Point) for p in [p1a, p1b, p1c]):
            return {"passed": False, "message": "Invalid point types in first angle"}
        
        angle1 = cmd.angle_ppp(p1a, p1b, p1c)
        
        # Get second angle
        points2 = points_list[1]
        if len(points2) != 3 or not all(p in element_dict for p in points2):
            return {"passed": False, "message": "Could not find second angle points"}
        
        p2a, p2b, p2c = [element_dict[p].data for p in points2]
        if not all(isinstance(p, gt.Point) for p in [p2a, p2b, p2c]):
            return {"passed": False, "message": "Invalid point types in second angle"}
        
        angle2 = cmd.angle_ppp(p2a, p2b, p2c)
        
        # Check equality
        result = cmd.are_congruent_aa(angle1, angle2)
        
        angle1_deg = np.degrees(angle1.angle)
        angle2_deg = np.degrees(angle2.angle)
        
        return {
            "passed": result.b,
            "message": f"Angles are {angle1_deg:.2f}° and {angle2_deg:.2f}°"
        }
    
    def _check_segment_equality(self, data: Dict) -> Dict[str, Any]:
        """Check if two segments are equal in length."""
        segments = data.get("segments", [])
        
        if len(segments) != 2:
            return {"passed": False, "message": "Segment equality requires 2 segments"}
        
        element_dict = self.construction.element_dict
        
        # Get segments
        seg1_points = segments[0]
        seg2_points = segments[1]
        
        if not all(p in element_dict for p in seg1_points + seg2_points):
            return {"passed": False, "message": "Could not find all points"}
        
        # Calculate distances
        p1a, p1b = [element_dict[p].data for p in seg1_points]
        p2a, p2b = [element_dict[p].data for p in seg2_points]
        
        if not all(isinstance(p, gt.Point) for p in [p1a, p1b, p2a, p2b]):
            return {"passed": False, "message": "Invalid point types"}
        
        dist1 = cmd.distance_pp(p1a, p1b)
        dist2 = cmd.distance_pp(p2a, p2b)
        
        result = cmd.are_equal_mm(dist1, dist2)
        
        return {
            "passed": result.b,
            "message": f"Segments have lengths {dist1.x:.2f} and {dist2.x:.2f}"
        }
    
    def _check_collinear(self, data: Dict) -> Dict[str, Any]:
        """Check if points are collinear."""
        points = data.get("points", [])
        
        if len(points) < 3:
            return {"passed": False, "message": "Collinearity requires at least 3 points"}
        
        element_dict = self.construction.element_dict
        
        # Check first 3 points (can extend to check all combinations)
        if not all(p in element_dict for p in points[:3]):
            return {"passed": False, "message": "Could not find all points"}
        
        p1, p2, p3 = [element_dict[p].data for p in points[:3]]
        
        if not all(isinstance(p, gt.Point) for p in [p1, p2, p3]):
            return {"passed": False, "message": "Invalid point types"}
        
        result = cmd.are_collinear_ppp(p1, p2, p3)
        
        return {
            "passed": result.b,
            "message": f"Points {'are' if result.b else 'are not'} collinear"
        }
    
    def _check_not_collinear(self, data: Dict) -> Dict[str, Any]:
        """Check if points are NOT collinear (for valid triangles)."""
        result = self._check_collinear(data)
        result["passed"] = not result["passed"]
        result["message"] = result["message"].replace("are collinear", "are not collinear").replace("are not not", "are")
        return result
    
    def _check_concyclic(self, data: Dict) -> Dict[str, Any]:
        """Check if four points lie on the same circle."""
        points = data.get("points", [])
        
        if len(points) != 4:
            return {"passed": False, "message": "Concyclic condition requires 4 points"}
        
        element_dict = self.construction.element_dict
        
        if not all(p in element_dict for p in points):
            return {"passed": False, "message": "Could not find all points"}
        
        p1, p2, p3, p4 = [element_dict[p].data for p in points]
        
        if not all(isinstance(p, gt.Point) for p in [p1, p2, p3, p4]):
            return {"passed": False, "message": "Invalid point types"}
        
        result = cmd.are_concyclic_pppp(p1, p2, p3, p4)
        
        return {
            "passed": result.b,
            "message": f"Points {'are' if result.b else 'are not'} concyclic"
        }
    
    def _check_concurrent(self, data: Dict) -> Dict[str, Any]:
        """Check if three lines meet at a point."""
        lines = data.get("lines", [])
        
        if len(lines) != 3:
            return {"passed": False, "message": "Concurrent condition requires 3 lines"}
        
        line_objs = []
        for line_points in lines:
            line = self._get_line_from_points(line_points[0], line_points[1])
            if line is None:
                return {"passed": False, "message": "Could not find all lines"}
            line_objs.append(line)
        
        result = cmd.are_concurrent_lll(*line_objs)
        
        return {
            "passed": result.b,
            "message": f"Lines {'are' if result.b else 'are not'} concurrent"
        }
    
    def _check_point_on_line(self, data: Dict) -> Dict[str, Any]:
        """Check if a point lies on a line."""
        point = data.get("point")
        line_points = data.get("line", [])
        
        if not point or len(line_points) != 2:
            return {"passed": False, "message": "Invalid point_on_line condition"}
        
        element_dict = self.construction.element_dict
        
        if point not in element_dict:
            return {"passed": False, "message": "Could not find point"}
        
        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": "Invalid point type"}
        
        line = self._get_line_from_points(line_points[0], line_points[1])
        if line is None:
            return {"passed": False, "message": "Could not find line"}
        
        result = cmd.contained_by_pl(pt, line)
        
        return {
            "passed": result.b,
            "message": f"Point {'is' if result.b else 'is not'} on line"
        }
    
    def _check_point_on_circle(self, data: Dict) -> Dict[str, Any]:
        """Check if a point lies on a circle."""
        point = data.get("point")
        circle_center = data.get("circle_center")
        
        if not point or not circle_center:
            return {"passed": False, "message": "Invalid point_on_circle condition"}
        
        element_dict = self.construction.element_dict
        
        if point not in element_dict or circle_center not in element_dict:
            return {"passed": False, "message": "Could not find point or circle"}
        
        pt = element_dict[point].data
        if not isinstance(pt, gt.Point):
            return {"passed": False, "message": "Invalid point type"}
        
        # Find circle with this center
        circle_label = self._find_circle_with_center(circle_center)
        if not circle_label:
            return {"passed": False, "message": "Could not find circle"}
        
        circle = element_dict[circle_label].data
        result = cmd.contained_by_pc(pt, circle)
        
        return {
            "passed": result.b,
            "message": f"Point {'is' if result.b else 'is not'} on circle"
        }
    
    def _check_angle_bisector(self, data: Dict) -> Dict[str, Any]:
        """Check if a line bisects an angle."""
        line_points = data.get("line", [])
        angle_points = data.get("angle_points", [])
        
        if len(line_points) != 2 or len(angle_points) != 3:
            return {"passed": False, "message": "Invalid angle_bisector condition"}
        
        element_dict = self.construction.element_dict
        
        # Get angle vertex (middle point)
        vertex = angle_points[1]
        if vertex not in element_dict:
            return {"passed": False, "message": "Could not find angle vertex"}
        
        # Check if bisector line passes through vertex
        if vertex not in line_points:
            # Check if vertex lies on the line
            line = self._get_line_from_points(line_points[0], line_points[1])
            vertex_pt = element_dict[vertex].data
            if not isinstance(vertex_pt, gt.Point) or not line.contains(vertex_pt.a):
                return {"passed": False, "message": "Bisector doesn't pass through angle vertex"}
        
        # Get all points
        if not all(p in element_dict for p in angle_points):
            return {"passed": False, "message": "Could not find angle points"}
        
        p1, vertex_pt, p3 = [element_dict[p].data for p in angle_points]
        
        if not all(isinstance(p, gt.Point) for p in [p1, vertex_pt, p3]):
            return {"passed": False, "message": "Invalid point types"}
        
        # Get a point on the bisector (not the vertex)
        bisector_pt = None
        for lp in line_points:
            if lp != vertex and lp in element_dict:
                bisector_pt = element_dict[lp].data
                break
        
        if bisector_pt is None or not isinstance(bisector_pt, gt.Point):
            return {"passed": False, "message": "Could not find bisector point"}
        
        # Calculate angles on both sides
        angle1 = cmd.angle_ppp(p1, vertex_pt, bisector_pt)
        angle2 = cmd.angle_ppp(bisector_pt, vertex_pt, p3)
        
        # Check if angles are equal
        result = cmd.are_congruent_aa(angle1, angle2)
        
        angle1_deg = np.degrees(angle1.angle)
        angle2_deg = np.degrees(angle2.angle)
        
        return {
            "passed": result.b,
            "message": f"Bisector creates angles of {angle1_deg:.2f}° and {angle2_deg:.2f}°"
        }


# Example usage
if __name__ == "__main__":
    from benchmark_dataset import BenchmarkDataset, RequiredObjects, ConditionBuilder
    
    # Test with an existing DSL file
    validator = DSLValidator()
    
    # Create a simple test problem
    required_objects = RequiredObjects(
        points=["A", "B", "C"],
        segments=[["A", "B"], ["B", "C"], ["A", "C"]],
        lines=[],
        circles=[],
        polygons=[["A", "B", "C"]]
    )
    
    from benchmark_dataset import BenchmarkProblem, VerificationCondition
    conditions = [
        VerificationCondition.from_dict(ConditionBuilder.not_collinear(["A", "B", "C"]))
    ]
    
    problem = BenchmarkProblem(
        id="test",
        subject="Test triangle",
        required_objects=required_objects,
        verification_conditions=conditions
    )
    
    print("DSL Validator created successfully")

