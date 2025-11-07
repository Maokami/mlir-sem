// Simple SCCP test - after optimization
// Constants have been propagated and folded

func.func @main() -> i64 {
  %result = arith.constant 130 : i64
  func.return %result : i64
}