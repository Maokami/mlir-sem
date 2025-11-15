# Building on Linux

As of the dune-configurator upgrade (see issue #36), the project detects your
platform and LLVM installation automatically. You no longer need to edit
`driver/dune` or `test/dune`; simply install LLVM and run:

```bash
opam exec -- dune build
opam exec -- dune test -f
```

Behind the scenes `config/detect_llvm.exe`:

- locates `LLVM_LIBDIR` by checking the `LLVM_LIBDIR` env var, `llvm-config
  --libdir`, and common distro paths,
- emits macOS-friendly `-Wl,-force_load` or Linux-friendly
  `-Wl,--whole-archive ... --no-whole-archive` flag sets, and
- selects the right C++ runtime (`lc++` vs `lstdc++`) plus MLIR/LLVM libraries.

## Manual overrides

Most distributions no longer need special setup, but you can still override the
detected path if LLVM is installed in a non-standard location:

```bash
export LLVM_LIBDIR=/path/to/custom/llvm/lib
opam exec -- dune build
```

If you are debugging the configurator, inspect
`config/mlir_link_flags.sexp` to see the precise flags it generated.

## Why detection still matters

macOS and Linux use different linker syntaxes for "whole archive" linking:

| Platform | Syntax | Description |
|----------|--------|-------------|
| **macOS** | `-Wl,-force_load,libfoo.a` | Per-file flag |
| **Linux** | `-Wl,--whole-archive libfoo.a -Wl,--no-whole-archive` | Range-based flag |

MLIR C API libraries rely on static initializers; without whole-archive linkage,
dialects never register and the driver crashes at runtime. On Linux our
configurator wraps the entire archive group automatically so that static
initializers are retained.

## Verification

After making changes, verify the build:

```bash
dune clean
dune build
dune test
```

Expected output:
```
Test Successful in X.XXs. 30 tests run.
```

## Troubleshooting

### Error: "cannot find -lMLIR" / missing archives

Set `LLVM_LIBDIR` explicitly to the directory that contains `libMLIR*.a`
and rerun `opam exec -- dune build`.

### Error: "unknown option: --whole-archive"

This should only appear if the configurator mis-detected your platform. Override
`LLVM_LIBDIR` to point at a macOS LLVM install or patch `config/detect_llvm.ml`
accordingly, then rerun the build.

## Related Issues

- [#36](https://github.com/Maokami/mlir-sem/issues/36): LLVM autodetection &
  Linux CI
- CI testing on Linux (active)

## See Also

- [LLVM CMake Documentation](https://llvm.org/docs/CMake.html)
- [macOS vs Linux Linking Differences](https://maskray.me/blog/2021-05-09-fno-semantic-interposition)
