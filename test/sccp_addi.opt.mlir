module {
  func.func @constant_prop_addi() -> i32 {
    %c30_i32 = arith.constant 30 : i32
    %c20_i32 = arith.constant 20 : i32
    %c10_i32 = arith.constant 10 : i32
    return %c30_i32 : i32
  }
}

