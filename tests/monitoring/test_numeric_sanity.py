"""
Tests for the Numeric Sanity detector.
"""

import pytest
from gpt_oss.monitoring.detectors.numeric_sanity import NumericSanityDetector


class TestNumericSanityDetector:
    """Test the NumericSanityDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = NumericSanityDetector()
        assert detector.tolerance == 0.01
        assert detector.unit_conversions is not None
        
        # Test custom tolerance
        detector = NumericSanityDetector(tolerance=0.1)
        assert detector.tolerance == 0.1
    
    def test_extract_numbers_and_units(self):
        """Test number and unit extraction."""
        detector = NumericSanityDetector()
        
        # Test basic number extraction
        text = "The distance is 100 km and the weight is 50 kg."
        numbers = detector._extract_numbers_and_units(text)
        
        assert len(numbers) == 2
        assert numbers[0]['value'] == 100.0
        assert numbers[0]['unit'] == 'km'
        assert numbers[1]['value'] == 50.0
        assert numbers[1]['unit'] == 'kg'
        
        # Test numbers without units
        text = "The temperature is 25 degrees and the price is 100 dollars."
        numbers = detector._extract_numbers_and_units(text)
        
        assert len(numbers) == 2
        assert numbers[0]['value'] == 25.0
        assert numbers[0]['unit'] is None
        assert numbers[1]['value'] == 100.0
        assert numbers[1]['unit'] is None
        
        # Test decimal numbers
        text = "The value is 3.14 and the ratio is 2.5."
        numbers = detector._extract_numbers_and_units(text)
        
        assert len(numbers) == 2
        assert numbers[0]['value'] == 3.14
        assert numbers[1]['value'] == 2.5
    
    def test_unit_conversion_checks(self):
        """Test unit conversion consistency checks."""
        detector = NumericSanityDetector()
        
        # Test consistent conversions
        assert detector._check_unit_conversion(100, 'km', 62.1371, 'mi')
        assert detector._check_unit_conversion(1, 'kg', 2.20462, 'lb')
        assert detector._check_unit_conversion(0, 'C', 32, 'F')
        
        # Test inconsistent conversions
        assert not detector._check_unit_conversion(100, 'km', 50, 'mi')  # Wrong conversion
        assert not detector._check_unit_conversion(1, 'kg', 1, 'lb')     # Wrong conversion
        
        # Test with no units
        assert detector._check_unit_conversion(100, None, 100, None)
        assert detector._check_unit_conversion(100, 'km', 100, None)
    
    def test_arithmetic_consistency(self):
        """Test arithmetic consistency checks."""
        detector = NumericSanityDetector()
        
        # Test with consistent arithmetic
        numbers = [
            {'value': 2, 'unit': None, 'start': 0, 'end': 1, 'text': '2'},
            {'value': 3, 'unit': None, 'start': 4, 'end': 5, 'text': '3'},
            {'value': 5, 'unit': None, 'start': 8, 'end': 9, 'text': '5'}
        ]
        
        issues = detector._check_arithmetic_consistency(numbers)
        # Should find that 2 + 3 = 5
        assert len(issues) > 0
        
        # Test with inconsistent arithmetic
        numbers = [
            {'value': 2, 'unit': None, 'start': 0, 'end': 1, 'text': '2'},
            {'value': 3, 'unit': None, 'start': 4, 'end': 5, 'text': '3'},
            {'value': 6, 'unit': None, 'start': 8, 'end': 9, 'text': '6'}  # Wrong sum
        ]
        
        issues = detector._check_arithmetic_consistency(numbers)
        # Should not find consistent arithmetic
        assert len(issues) == 0
    
    def test_unit_consistency(self):
        """Test unit consistency checks."""
        detector = NumericSanityDetector()
        
        # Test with consistent units
        numbers = [
            {'value': 100, 'unit': 'km', 'start': 0, 'end': 6, 'text': '100 km'},
            {'value': 62.1371, 'unit': 'mi', 'start': 8, 'end': 16, 'text': '62.1371 mi'}
        ]
        
        issues = detector._check_unit_consistency(numbers)
        # Should be consistent
        assert len(issues) == 0
        
        # Test with inconsistent units
        numbers = [
            {'value': 100, 'unit': 'km', 'start': 0, 'end': 6, 'text': '100 km'},
            {'value': 50, 'unit': 'mi', 'start': 8, 'end': 12, 'text': '50 mi'}  # Wrong conversion
        ]
        
        issues = detector._check_unit_consistency(numbers)
        # Should find inconsistency
        assert len(issues) > 0
    
    def test_detect_basic(self):
        """Test basic detection functionality."""
        detector = NumericSanityDetector()
        
        # Test with consistent text
        text = "The distance is 100 km which equals 62.1371 miles."
        score, issues = detector.detect(text)
        
        assert 0 <= score <= 1
        assert isinstance(issues, list)
        
        # Test with inconsistent text
        text = "The distance is 100 km which equals 50 miles."  # Wrong conversion
        score, issues = detector.detect(text)
        
        assert 0 <= score <= 1
        assert isinstance(issues, list)
    
    def test_detect_no_numbers(self):
        """Test detection with no numbers."""
        detector = NumericSanityDetector()
        
        text = "This text contains no numbers at all."
        score, issues = detector.detect(text)
        
        assert score == 1.0  # Perfect score when no numbers
        assert len(issues) == 0
    
    def test_detect_complex_scenario(self):
        """Test detection with complex scenario."""
        detector = NumericSanityDetector()
        
        # Text with multiple numeric issues
        text = """
        The temperature is 25°C which is 77°F. 
        The distance is 100 km which equals 50 miles (wrong!).
        The weight is 1 kg which is 2.2 lb.
        The sum of 2 and 3 is 6 (wrong!).
        """
        
        score, issues = detector.detect(text)
        
        assert 0 <= score <= 1
        assert isinstance(issues, list)
        
        # Should detect some issues
        assert len(issues) > 0
    
    def test_temperature_conversions(self):
        """Test temperature conversion logic."""
        detector = NumericSanityDetector()
        
        # Test C to F conversion
        assert detector._check_unit_conversion(0, 'C', 32, 'F')
        assert detector._check_unit_conversion(100, 'C', 212, 'F')
        assert detector._check_unit_conversion(25, 'C', 77, 'F')
        
        # Test F to C conversion
        assert detector._check_unit_conversion(32, 'F', 0, 'C')
        assert detector._check_unit_conversion(212, 'F', 100, 'C')
        assert detector._check_unit_conversion(77, 'F', 25, 'C')
        
        # Test wrong conversions
        assert not detector._check_unit_conversion(0, 'C', 0, 'F')
        assert not detector._check_unit_conversion(100, 'C', 100, 'F')
    
    def test_distance_conversions(self):
        """Test distance conversion logic."""
        detector = NumericSanityDetector()
        
        # Test km to mi conversion
        assert detector._check_unit_conversion(1, 'km', 0.621371, 'mi')
        assert detector._check_unit_conversion(100, 'km', 62.1371, 'mi')
        
        # Test mi to km conversion
        assert detector._check_unit_conversion(1, 'mi', 1.60934, 'km')
        assert detector._check_unit_conversion(62.1371, 'mi', 100, 'km')
        
        # Test wrong conversions
        assert not detector._check_unit_conversion(1, 'km', 1, 'mi')
        assert not detector._check_unit_conversion(100, 'km', 100, 'mi')
    
    def test_weight_conversions(self):
        """Test weight conversion logic."""
        detector = NumericSanityDetector()
        
        # Test kg to lb conversion
        assert detector._check_unit_conversion(1, 'kg', 2.20462, 'lb')
        assert detector._check_unit_conversion(10, 'kg', 22.0462, 'lb')
        
        # Test lb to kg conversion
        assert detector._check_unit_conversion(1, 'lb', 0.453592, 'kg')
        assert detector._check_unit_conversion(22.0462, 'lb', 10, 'kg')
        
        # Test wrong conversions
        assert not detector._check_unit_conversion(1, 'kg', 1, 'lb')
        assert not detector._check_unit_conversion(10, 'kg', 10, 'lb')
    
    def test_tolerance_handling(self):
        """Test tolerance handling in conversions."""
        detector = NumericSanityDetector(tolerance=0.1)
        
        # Test with higher tolerance
        assert detector._check_unit_conversion(100, 'km', 62, 'mi')  # Within 0.1 tolerance
        assert detector._check_unit_conversion(100, 'km', 63, 'mi')  # Within 0.1 tolerance
        
        # Test with lower tolerance
        detector_low_tolerance = NumericSanityDetector(tolerance=0.001)
        assert not detector_low_tolerance._check_unit_conversion(100, 'km', 62, 'mi')  # Outside tolerance
