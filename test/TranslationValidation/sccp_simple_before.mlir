// Simple SCCP test - before optimization
// This function has opportunities for constant propagation

func.func @main() -> i64 {
  %c1 = arith.constant 10 : i64
  %c2 = arith.constant 20 : i64
  %x = arith.addi %c1, %c2 : i64    // x = 30 (can be folded)
  %c3 = arith.constant 100 : i64
  %y = arith.addi %x, %c3 : i64     // y = 130 (can be folded)
  func.return %y : i64
}