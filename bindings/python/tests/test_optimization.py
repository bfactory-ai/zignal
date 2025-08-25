"""Test optimization module functionality."""

import numpy as np
import pytest
import zignal


def test_optimization_policy_enum():
    """Test that OptimizationPolicy enum is accessible and has expected values."""
    assert hasattr(zignal, "OptimizationPolicy")
    assert hasattr(zignal.OptimizationPolicy, "MIN")
    assert hasattr(zignal.OptimizationPolicy, "MAX")
    assert zignal.OptimizationPolicy.MIN.value == 0
    assert zignal.OptimizationPolicy.MAX.value == 1


def test_assignment_type():
    """Test that Assignment type is accessible."""
    assert hasattr(zignal, "Assignment")


def test_solve_assignment_problem_basic():
    """Test basic assignment problem solving."""
    # Create a simple 3x3 cost matrix
    costs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    matrix = zignal.Matrix.from_numpy(costs)

    # Solve for minimum cost
    result = zignal.solve_assignment_problem(matrix)

    # Check result type
    assert isinstance(result, zignal.Assignment)
    assert hasattr(result, "assignments")
    assert hasattr(result, "total_cost")

    # Check assignments
    assert len(result.assignments) == 3
    assert all(x is None or isinstance(x, int) for x in result.assignments)
    assert all(x is None or 0 <= x < 3 for x in result.assignments)

    # Check that total cost is reasonable
    assert isinstance(result.total_cost, float)
    assert result.total_cost >= 0


def test_solve_assignment_problem_minimize():
    """Test minimization policy."""
    # Create a cost matrix where diagonal is cheapest
    costs = np.array([[1.0, 10.0, 10.0], [10.0, 2.0, 10.0], [10.0, 10.0, 3.0]], dtype=np.float64)
    matrix = zignal.Matrix.from_numpy(costs)

    # Solve for minimum cost
    result = zignal.solve_assignment_problem(matrix, zignal.OptimizationPolicy.MIN)

    # Optimal should be diagonal (0->0, 1->1, 2->2) with cost 1+2+3=6
    assert result.total_cost == pytest.approx(6.0)
    assert result.assignments == [0, 1, 2]


def test_solve_assignment_problem_maximize():
    """Test maximization policy."""
    # Create a profit matrix where anti-diagonal is most profitable
    profits = np.array([[1.0, 2.0, 10.0], [2.0, 5.0, 8.0], [10.0, 6.0, 3.0]], dtype=np.float64)
    matrix = zignal.Matrix.from_numpy(profits)

    # Solve for maximum profit
    result = zignal.solve_assignment_problem(matrix, zignal.OptimizationPolicy.MAX)

    # Check that we get a valid assignment
    assert len(result.assignments) == 3
    assert result.total_cost > 0  # Should be positive for profits

    # The maximum should be at least 10+8+6=24 (one possible optimal)
    assert result.total_cost >= 24.0


def test_solve_assignment_problem_rectangular():
    """Test with rectangular matrices."""
    # Test 2x3 matrix (more columns than rows)
    costs = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 1.0]], dtype=np.float64)
    matrix = zignal.Matrix.from_numpy(costs)

    result = zignal.solve_assignment_problem(matrix)

    # Should have 2 assignments (one for each row)
    assert len(result.assignments) == 2
    assert all(x is None or 0 <= x < 3 for x in result.assignments)

    # Check that assigned columns are unique (if both are assigned)
    assigned_cols = [x for x in result.assignments if x is not None]
    assert len(assigned_cols) == len(set(assigned_cols))  # No duplicates


def test_solve_assignment_problem_rectangular_tall():
    """Test with tall rectangular matrix (more rows than columns)."""
    # Test 3x2 matrix
    costs = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    matrix = zignal.Matrix.from_numpy(costs)

    result = zignal.solve_assignment_problem(matrix)

    # Should have 3 potential assignments (one for each row)
    assert len(result.assignments) == 3

    # At most 2 rows can be assigned (only 2 columns available)
    assigned_count = sum(1 for x in result.assignments if x is not None)
    assert assigned_count <= 2


def test_solve_assignment_problem_single_element():
    """Test with 1x1 matrix."""
    costs = np.array([[5.0]], dtype=np.float64)
    matrix = zignal.Matrix.from_numpy(costs)

    result = zignal.solve_assignment_problem(matrix)

    assert len(result.assignments) == 1
    assert result.assignments[0] == 0
    assert result.total_cost == pytest.approx(5.0)


def test_solve_assignment_problem_integer_costs():
    """Test that integer costs work correctly."""
    # Create matrix with integer values
    costs = np.array(
        [[10, 20, 30], [15, 25, 35], [20, 30, 40]],
        dtype=np.float64,  # Matrix expects float64
    )
    matrix = zignal.Matrix.from_numpy(costs)

    result = zignal.solve_assignment_problem(matrix)

    # Should get valid assignments
    assert len(result.assignments) == 3
    assert isinstance(result.total_cost, float)
    assert result.total_cost > 0


def test_solve_assignment_problem_zeros():
    """Test with matrix containing zeros."""
    costs = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]], dtype=np.float64)
    matrix = zignal.Matrix.from_numpy(costs)

    result = zignal.solve_assignment_problem(matrix)

    # Optimal is all zeros on diagonal, total cost = 0
    assert result.total_cost == pytest.approx(0.0)


def test_assignment_repr():
    """Test Assignment object string representation."""
    costs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    matrix = zignal.Matrix.from_numpy(costs)
    result = zignal.solve_assignment_problem(matrix)

    repr_str = repr(result)
    assert "Assignment" in repr_str
    assert "total_cost" in repr_str


def test_invalid_policy():
    """Test that invalid policy values are rejected."""
    costs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    matrix = zignal.Matrix.from_numpy(costs)

    # String values should be rejected
    with pytest.raises(TypeError):
        zignal.solve_assignment_problem(matrix, "invalid")

    # Raw ints 0 and 1 are allowed (they match enum values)
    result = zignal.solve_assignment_problem(matrix, 0)  # MIN
    assert isinstance(result, zignal.Assignment)

    result = zignal.solve_assignment_problem(matrix, 1)  # MAX
    assert isinstance(result, zignal.Assignment)

    # Invalid integer values should be rejected
    with pytest.raises(ValueError):
        zignal.solve_assignment_problem(matrix, 2)  # Invalid enum value


def test_invalid_matrix_type():
    """Test that non-Matrix inputs are rejected."""
    costs = [[1.0, 2.0], [3.0, 4.0]]

    with pytest.raises(TypeError):
        zignal.solve_assignment_problem(costs)

    # NumPy array directly should also fail
    np_costs = np.array(costs)
    with pytest.raises(TypeError):
        zignal.solve_assignment_problem(np_costs)
