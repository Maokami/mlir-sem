func.func @main() -> (i32) {
  %0 = arith.constant 21 : i32
  %1 = arith.addi %0, %0 : i32
  func.return %1 : i32
}