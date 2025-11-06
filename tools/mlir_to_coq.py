#!/usr/bin/env python3
"""
MLIR to Coq AST converter for Translation Validation

This script converts MLIR programs to Coq definitions that can be used
for translation validation proofs.
"""

import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

class MLIRToCoq:
    """Converter from MLIR text format to Coq AST definitions."""

    def __init__(self):
        self.ssa_counter = 0
        self.block_counter = 0
        self.ssa_map: Dict[str, int] = {}

    def reset(self):
        """Reset internal state for new conversion."""
        self.ssa_counter = 0
        self.block_counter = 0
        self.ssa_map = {}

    def get_ssa_id(self, mlir_ssa: str) -> int:
        """Convert MLIR SSA value name to numeric ID."""
        if mlir_ssa not in self.ssa_map:
            self.ssa_map[mlir_ssa] = self.ssa_counter
            self.ssa_counter += 1
        return self.ssa_map[mlir_ssa]

    def parse_constant(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse arith.constant operation."""
        match = re.match(r'\s*(%\w+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*i\d+', line)
        if match:
            ssa_name = match.group(1)
            value = match.group(2)
            ssa_id = self.get_ssa_id(ssa_name)
            return f"OpConstant {value}", f"(* {ssa_name} = {value} *)"
        return None

    def parse_addi(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse arith.addi operation."""
        match = re.match(r'\s*(%\w+)\s*=\s*arith\.addi\s+(%\w+),\s*(%\w+)\s*:\s*i\d+', line)
        if match:
            result = match.group(1)
            op1 = match.group(2)
            op2 = match.group(3)

            result_id = self.get_ssa_id(result)
            op1_id = self.get_ssa_id(op1)
            op2_id = self.get_ssa_id(op2)

            return (f"OpAddi (SSAVal {op1_id}) (SSAVal {op2_id})",
                   f"(* {result} = {op1} + {op2} *)")
        return None

    def parse_return(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse return statement."""
        match = re.match(r'\s*return\s+(%\w+)\s*:\s*i\d+', line)
        if match:
            ssa_name = match.group(1)
            ssa_id = self.get_ssa_id(ssa_name)
            return f"TermReturn (SSAVal {ssa_id})", f"(* return {ssa_name} *)"
        return None

    def parse_cond_branch(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse cf.cond_br statement."""
        match = re.match(r'\s*cf\.cond_br\s+(%\w+),\s*\^bb(\d+),\s*\^bb(\d+)', line)
        if match:
            cond = match.group(1)
            true_bb = match.group(2)
            false_bb = match.group(3)

            cond_id = self.get_ssa_id(cond)
            return (f"TermCondBranch (SSAVal {cond_id}) {true_bb} {false_bb}",
                   f"(* cond_br {cond}, ^bb{true_bb}, ^bb{false_bb} *)")
        return None

    def parse_branch(self, line: str) -> Optional[Tuple[str, str]]:
        """Parse cf.br statement."""
        match = re.match(r'\s*cf\.br\s+\^bb(\d+)', line)
        if match:
            target = match.group(1)
            return f"TermBranch {target}", f"(* br ^bb{target} *)"
        return None

    def parse_mlir(self, mlir_text: str) -> str:
        """Convert MLIR text to Coq AST definition."""
        self.reset()
        lines = mlir_text.split('\n')

        # Extract function name
        func_match = re.search(r'func\.func\s+@(\w+)', mlir_text)
        func_name = func_match.group(1) if func_match else "unknown"

        # Parse operations
        operations = []
        terminator = None

        for line in lines:
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('//'):
                continue

            # Try parsing different instruction types
            if 'arith.constant' in line:
                result = self.parse_constant(line)
                if result:
                    operations.append(result)
            elif 'arith.addi' in line:
                result = self.parse_addi(line)
                if result:
                    operations.append(result)
            elif 'return' in line:
                terminator = self.parse_return(line)
            elif 'cf.cond_br' in line:
                terminator = self.parse_cond_branch(line)
            elif 'cf.br' in line:
                terminator = self.parse_branch(line)

        # Generate Coq definition
        coq_def = f"Definition {func_name}_program : mlir_program :=\n"

        # Operations
        if operations:
            coq_def += "  let ops := [\n"
            for i, (op, comment) in enumerate(operations):
                semicolon = ";" if i < len(operations) - 1 else ""
                coq_def += f"    {op}{semicolon} {comment}\n"
            coq_def += "  ] in\n"

        # Block
        coq_def += "  let main_block := {|\n"
        coq_def += "    block_label := 0;\n"
        coq_def += f"    block_ops := {('ops' if operations else '[]')};\n"

        if terminator:
            op_code, comment = terminator
            coq_def += f"    block_terminator := {op_code} {comment}\n"
        else:
            coq_def += "    block_terminator := TermReturn (SSAVal 0) (* default *)\n"

        coq_def += "  |} in\n"

        # Function
        coq_def += "  let main_func := {|\n"
        coq_def += f'    func_name := "{func_name}";\n'
        coq_def += "    func_type := FuncType [] I32;\n"
        coq_def += "    func_args := [];\n"
        coq_def += "    func_blocks := [main_block]\n"
        coq_def += "  |} in\n"

        # Program
        coq_def += "  {| prog_funcs := [main_func] |}.\n"

        return coq_def

def main():
    parser = argparse.ArgumentParser(description='Convert MLIR to Coq AST')
    parser.add_argument('input', help='Input MLIR file')
    parser.add_argument('-o', '--output', help='Output Coq file')
    parser.add_argument('--name', help='Name for the Coq definition')

    args = parser.parse_args()

    # Read input
    with open(args.input, 'r') as f:
        mlir_text = f.read()

    # Convert
    converter = MLIRToCoq()
    coq_def = converter.parse_mlir(mlir_text)

    # Customize name if provided
    if args.name:
        coq_def = coq_def.replace('_program :', f'_{args.name} :')

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(coq_def)
        print(f"Converted {args.input} to {args.output}")
    else:
        print(coq_def)

if __name__ == '__main__':
    main()