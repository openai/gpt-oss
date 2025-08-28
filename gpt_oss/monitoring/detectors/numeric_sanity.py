"""
Numeric Sanity detector for hallucination monitoring.

Detects numbers and units, checks arithmetic consistency and unit conversions.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math

logger = logging.getLogger(__name__)


class NumericSanityDetector:
    """Detector for numeric sanity and consistency."""
    
    def __init__(self, tolerance: float = 0.01):
        """Initialize the detector with tolerance for arithmetic checks."""
        self.tolerance = tolerance
        
        # Common unit conversion factors
        self.unit_conversions = {
            # Distance
            ('km', 'mi'): 0.621371,
            ('mi', 'km'): 1.60934,
            ('m', 'ft'): 3.28084,
            ('ft', 'm'): 0.3048,
            ('cm', 'in'): 0.393701,
            ('in', 'cm'): 2.54,
            
            # Weight
            ('kg', 'lb'): 2.20462,
            ('lb', 'kg'): 0.453592,
            ('g', 'oz'): 0.035274,
            ('oz', 'g'): 28.3495,
            
            # Temperature
            ('C', 'F'): lambda c: c * 9/5 + 32,
            ('F', 'C'): lambda f: (f - 32) * 5/9,
            
            # Volume
            ('L', 'gal'): 0.264172,
            ('gal', 'L'): 3.78541,
            ('ml', 'fl_oz'): 0.033814,
            ('fl_oz', 'ml'): 29.5735,
        }
    
    def _extract_numbers_and_units(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbers and their associated units from text."""
        # Pattern to match numbers with optional units
        number_pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z%°]+)?'
        
        matches = []
        for match in re.finditer(number_pattern, text):
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else None
            
            matches.append({
                'value': value,
                'unit': unit,
                'start': match.start(),
                'end': match.end(),
                'text': match.group(0)
            })
        
        return matches
    
    def _check_unit_conversion(self, value1: float, unit1: str, value2: float, unit2: str) -> bool:
        """Check if two values with different units are consistent."""
        if not unit1 or not unit2:
            return True  # No units to compare
        
        # Normalize units to lowercase
        unit1, unit2 = unit1.lower(), unit2.lower()
        
        # Check if we have a conversion factor
        if (unit1, unit2) in self.unit_conversions:
            factor = self.unit_conversions[(unit1, unit2)]
            if callable(factor):
                converted = factor(value1)
            else:
                converted = value1 * factor
            
            return abs(converted - value2) <= (self.tolerance * max(abs(converted), abs(value2)))
        
        elif (unit2, unit1) in self.unit_conversions:
            factor = self.unit_conversions[(unit2, unit1)]
            if callable(factor):
                converted = factor(value2)
            else:
                converted = value2 * factor
            
            return abs(converted - value1) <= (self.tolerance * max(abs(converted), abs(value1)))
        
        return True  # No conversion available, assume consistent
    
    def _check_arithmetic_consistency(self, numbers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for simple arithmetic relationships between numbers."""
        issues = []
        
        for i, num1 in enumerate(numbers):
            for j, num2 in enumerate(numbers[i+1:], i+1):
                # Check for simple relationships
                val1, val2 = num1['value'], num2['value']
                
                # Skip if units are incompatible
                if num1['unit'] and num2['unit'] and num1['unit'] != num2['unit']:
                    continue
                
                # Check for sum relationship (e.g., "3 + 4 = 7")
                if abs(val1 + val2 - (val1 + val2)) > self.tolerance:
                    # Look for a third number that might be the sum
                    for k, num3 in enumerate(numbers):
                        if k != i and k != j:
                            if abs(num3['value'] - (val1 + val2)) <= self.tolerance:
                                issues.append({
                                    'type': 'arithmetic_sum',
                                    'numbers': [num1, num2, num3],
                                    'expected': val1 + val2,
                                    'found': num3['value'],
                                    'consistent': True
                                })
                
                # Check for product relationship
                if abs(val1 * val2 - (val1 * val2)) > self.tolerance:
                    for k, num3 in enumerate(numbers):
                        if k != i and k != j:
                            if abs(num3['value'] - (val1 * val2)) <= self.tolerance:
                                issues.append({
                                    'type': 'arithmetic_product',
                                    'numbers': [num1, num2, num3],
                                    'expected': val1 * val2,
                                    'found': num3['value'],
                                    'consistent': True
                                })
        
        return issues
    
    def _check_unit_consistency(self, numbers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for unit consistency across related numbers."""
        issues = []
        
        # Group numbers by similar contexts (e.g., distances, weights)
        distance_units = {'km', 'mi', 'm', 'ft', 'cm', 'in'}
        weight_units = {'kg', 'lb', 'g', 'oz'}
        temperature_units = {'C', 'F', '°C', '°F'}
        
        unit_groups = {
            'distance': distance_units,
            'weight': weight_units,
            'temperature': temperature_units
        }
        
        for group_name, units in unit_groups.items():
            group_numbers = [n for n in numbers if n['unit'] and n['unit'].lower() in units]
            
            if len(group_numbers) >= 2:
                # Check conversions between different units in the same group
                for i, num1 in enumerate(group_numbers):
                    for num2 in group_numbers[i+1:]:
                        if num1['unit'] != num2['unit']:
                            consistent = self._check_unit_conversion(
                                num1['value'], num1['unit'], 
                                num2['value'], num2['unit']
                            )
                            
                            if not consistent:
                                issues.append({
                                    'type': 'unit_inconsistency',
                                    'numbers': [num1, num2],
                                    'group': group_name,
                                    'consistent': False
                                })
        
        return issues
    
    def detect(self, text: str) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Detect numeric sanity issues in text.
        
        Returns:
            - sanity_score: float in [0, 1]
            - issues: List of detected numeric issues
        """
        logger.info("Running numeric sanity detection")
        
        # Extract numbers and units
        numbers = self._extract_numbers_and_units(text)
        
        if not numbers:
            logger.info("No numbers found in text")
            return 1.0, []
        
        # Check for issues
        arithmetic_issues = self._check_arithmetic_consistency(numbers)
        unit_issues = self._check_unit_consistency(numbers)
        
        all_issues = arithmetic_issues + unit_issues
        
        # Count inconsistent issues
        inconsistent_count = sum(1 for issue in all_issues if not issue.get('consistent', True))
        total_checks = len(all_issues) if all_issues else 1
        
        # Compute sanity score
        sanity_score = 1.0 - (inconsistent_count / total_checks)
        
        logger.info(f"Numeric sanity score: {sanity_score:.3f} ({inconsistent_count}/{total_checks} issues)")
        
        return sanity_score, all_issues
