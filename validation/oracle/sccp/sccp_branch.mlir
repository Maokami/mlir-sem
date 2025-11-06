// Test SCCP with constant conditional branch
func.func @constant_branch(%arg0: i32) -> i32 {
  %true = arith.constant true
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  cf.cond_br %true, ^bb1, ^bb2

^bb1:
  return %c1 : i32

^bb2:
  return %c2 : i32
}
