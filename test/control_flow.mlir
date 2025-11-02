func.func @main() -> i32 {
  // Entry block: branches unconditionally to ^block_b
  cf.br ^block_b

^block_a:
  // This block should not be executed.
  %val_a = arith.constant 99 : i32
  func.return %val_a : i32

^block_b:
  // This block should be executed.
  %val_b = arith.constant 42 : i32
  func.return %val_b : i32
}
