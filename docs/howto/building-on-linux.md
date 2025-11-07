# Building on Linux

The project's `driver/dune` and `test/dune` files currently use macOS-specific linker flags (`-Wl,-force_load`). To build on Linux, you need to modify these files to use Linux-compatible flags.

## Quick Fix

### 1. Edit `driver/dune`

Find all occurrences of:
```
-Wl,-force_load,%{env:LLVM_LIBDIR=/usr/local/opt/llvm/lib}/libMLIRCAPI*.a
```

Replace with the Linux equivalent by wrapping all libraries:

**Before** (macOS):
```scheme
-cclib
"-Wl,-force_load,%{env:LLVM_LIBDIR=/usr/local/opt/llvm/lib}/libMLIRCAPIIR.a"
-cclib
"-Wl,-force_load,%{env:LLVM_LIBDIR=/usr/local/opt/llvm/lib}/libMLIRCAPIFunc.a"
...
```

**After** (Linux):
```scheme
-cclib
"-Wl,--whole-archive"
-cclib
"%{env:LLVM_LIBDIR=/usr/lib/llvm-18/lib}/libMLIRCAPIIR.a"
-cclib
"%{env:LLVM_LIBDIR=/usr/lib/llvm-18/lib}/libMLIRCAPIFunc.a"
-cclib
"%{env:LLVM_LIBDIR=/usr/lib/llvm-18/lib}/libMLIRCAPIArith.a"
-cclib
"%{env:LLVM_LIBDIR=/usr/lib/llvm-18/lib}/libMLIRCAPIControlFlow.a"
-cclib
"%{env:LLVM_LIBDIR=/usr/lib/llvm-18/lib}/libMLIRCAPIRegisterEverything.a"
-cclib
"-Wl,--no-whole-archive"
```

**Key differences**:
- Use `--whole-archive` / `--no-whole-archive` to wrap library group
- Remove `-force_load` prefix from each library
- Update `LLVM_LIBDIR` default path for your Linux distribution

### 2. Edit `test/dune`

Apply the same changes to the `test_driver` test configuration.

### 3. Set Environment Variables

Linux typically installs LLVM in different locations:

```bash
# Ubuntu/Debian
export LLVM_LIBDIR=/usr/lib/llvm-18/lib

# Fedora/RHEL
export LLVM_LIBDIR=/usr/lib64/llvm

# Custom installation
export LLVM_LIBDIR=/path/to/llvm/lib
```

## Why This is Needed

macOS and Linux use different linker syntaxes for "whole archive" linking:

| Platform | Syntax | Description |
|----------|--------|-------------|
| **macOS** | `-Wl,-force_load,libfoo.a` | Per-file flag |
| **Linux** | `-Wl,--whole-archive libfoo.a -Wl,--no-whole-archive` | Range-based flag |

The MLIR C API libraries require whole-archive linking because they register dialects via static initializers that would otherwise be optimized away by the linker.

## Automated Solution (Planned)

We plan to use `dune-configurator` to detect the platform automatically and generate appropriate flags. Track progress in the related issue.

For now, manual modification is required for Linux builds.

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

### Error: "unknown option: --whole-archive"

**Cause**: You're on macOS but using Linux flags.

**Solution**: Use the macOS syntax with `-force_load`.

### Error: "undefined reference to mlirContextCreate"

**Cause**: MLIR libraries are not being linked with whole-archive.

**Solution**: Verify `--whole-archive` wraps **all** MLIR C API libraries.

### Error: "cannot find -lMLIR"

**Cause**: LLVM_LIBDIR environment variable is incorrect.

**Solution**: Find your LLVM installation and set `LLVM_LIBDIR`:
```bash
find /usr -name "libMLIR.so" 2>/dev/null
export LLVM_LIBDIR=/path/to/directory/containing/libMLIR.so
```

## Related Issues

- Platform detection for linker flags (planned)
- CI testing on Linux (planned)

## See Also

- [LLVM CMake Documentation](https://llvm.org/docs/CMake.html)
- [macOS vs Linux Linking Differences](https://maskray.me/blog/2021-05-09-fno-semantic-interposition)
