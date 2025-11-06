// Test SCCP constant propagation with addi
func.func @constant_prop_addi() -> i32 {
  %c1 = arith.constant 10 : i32
  %c2 = arith.constant 20 : i32
  %result = arith.addi %c1, %c2 : i32
  return %result : i32
}
