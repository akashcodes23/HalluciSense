"""Phase 42 — Symbolic Arithmetic Verifier.

Executes deterministic mathematical evaluation using a safe AST parser.
Strictly zero eval() or arbitrary code execution.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Dict, Optional


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


class SafeArithmeticEvaluator(ast.NodeVisitor):
    """AST visitor that evaluates basic arithmetic without allowing function calls or imports."""

    def visit(self, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        elif isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                    raise ZeroDivisionError("Division by zero in arithmetic claim")
                return SAFE_OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            op_type = type(node.op)
            if op_type in SAFE_OPERATORS:
                return SAFE_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        else:
            raise ValueError(f"Unsafe or unsupported AST node: {type(node)}")


def evaluate_arithmetic_claim(claim_text: str) -> Optional[Dict[str, Any]]:
    """Deterministically parse and evaluate an arithmetic claim."""
    text = claim_text.strip().lower()
    
    # Normalize natural language operators
    normalized = text.replace("multiplied by", "*").replace("times", "*").replace("x", "*").replace("×", "*")
    normalized = normalized.replace("divided by", "/").replace("÷", "/")
    normalized = normalized.replace("plus", "+").replace("minus", "-")
    normalized = normalized.replace("equals", "=").replace("is equal to", "=").replace("is", "=")
    
    # Extract percentage if present
    if "%" in normalized and "of" in normalized:
        # e.g., 15% of 200 = 30 -> (15/100) * 200 = 30
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)", normalized)
        if pct_match:
            pct_val = float(pct_match.group(1))
            base_val = float(pct_match.group(2))
            claimed_val = float(pct_match.group(3))
            computed_val = (pct_val / 100.0) * base_val
            consistent = abs(computed_val - claimed_val) < 1e-4
            return {
                "verified": True,
                "operation": "percentage",
                "computed_value": round(computed_val, 4),
                "claimed_value": round(claimed_val, 4),
                "is_consistent": consistent,
                "explanation": f"{pct_val}% of {base_val} is {computed_val:.2f} (Claim stated {claimed_val:.2f})",
            }
            
    if "=" not in normalized:
        return None
        
    parts = normalized.split("=")
    if len(parts) != 2:
        return None
        
    expr_str = parts[0].strip()
    claimed_str = parts[1].strip()
    
    try:
        # Parse expression safely
        tree = ast.parse(expr_str, mode="eval")
        evaluator = SafeArithmeticEvaluator()
        computed_val = evaluator.visit(tree)
        
        # Parse claimed value
        claimed_tree = ast.parse(claimed_str, mode="eval")
        claimed_val = evaluator.visit(claimed_tree)
        
        consistent = abs(computed_val - claimed_val) < 1e-4
        return {
            "verified": True,
            "operation": "arithmetic",
            "computed_value": round(computed_val, 4),
            "claimed_value": round(claimed_val, 4),
            "is_consistent": consistent,
            "explanation": f"Computed {expr_str} = {computed_val:.4f} (Claim stated {claimed_val:.4f})",
        }
    except Exception as e:
        return {
            "verified": False,
            "operation": "arithmetic_parse_error",
            "error": str(e),
            "is_consistent": None,
        }
