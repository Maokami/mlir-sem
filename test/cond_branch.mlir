func.func @main() -> i32 {
  %c1_main = arith.constant 1 : i32
  %c0_main = arith.constant 0 : i32
  %true_val = arith.constant 10 : i32
  %false_val = arith.constant 20 : i32
  // if 1 != 0 goto ^true_block(%true_val) else ^false_block(%false_val)
  %cond = arith.cmpi ne, %c1_main, %c0_main : i32
  cf.cond_br %cond, ^true_block(%true_val : i32), ^false_block(%false_val : i32)

^true_block(%arg_true : i32):
  // should return 10 + 5 = 15
  %c5_true_block = arith.constant 5 : i32
  %res_true = arith.addi %arg_true, %c5_true_block : i32
  cf.br ^exit_block(%res_true : i32)

^false_block(%arg_false : i32):
  // should not be taken
  %c1_false_block = arith.constant 1 : i32
  %res_false = arith.addi %arg_false, %c1_false_block : i32
  cf.br ^exit_block(%res_false : i32)

^exit_block(%final_val_exit : i32):
  func.return %final_val_exit : i32
}
