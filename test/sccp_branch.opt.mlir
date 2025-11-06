module {
  func.func @constant_branch(%arg0: i32) -> i32 {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    cf.cond_br %true, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    return %c1_i32 : i32
  ^bb2:  // pred: ^bb0
    return %c2_i32 : i32
  }
}

