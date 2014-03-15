Rusty Kaleidoscope
==================
An implementation of LLVM's wonderful Kaleidoscope tutorial in Rust. Uses
`librustc`'s LLVM binding because it exists.

Compilation
-----------
First things first, add the following to `rust/src/rustllvm/RustWrapper.cpp`:
```
extern "C" bool
LLVMRustInitializeNativeTarget() {
  // JIT/native target initialization is handled strangely at best
  return LLVMInitializeNativeTarget();
}
```
After adding this wrapper function, re-run `make all` in your source directory.

You can then build lang.rs with this invocation:
```
RUST_OUTDIR=<path to rust/build-triple, i.e. rust/x86_64-unknown-linux-gnu
rustc lang.rs -g -L $RUST_OUTDIR/llvm/Release+Asserts/lib/ -L $RUST_OUTDIR/rt/
```

Done
----
- Parsing
- Code generation
- Error detection
- JIT compilation/evaluation

Todo
----
- Better error recovery
- Exercise 4 onward
- Ditch dependence on `librustc` and rust's source directory

Example
-------
```
ready> def a(b) b*b;
Parsed a function definition

define double @a(double %b) {
entry:
  %multmp = fmul double %b, %b
  ret double %multmp
}

ready> a(5);
Parsed a top level expr

define double @0() {
entry:
  %calltmp = call double @a(double 5.000000e+00)
  ret double %calltmp
}

Executing function
Returned 25
ready> def c(x) a(x+2);
Parsed a function definition

define double @c(double %x) {
entry:
  %addtmp = fadd double 2.000000e+00, %x
  %calltmp = call double @a(double %addtmp)
  ret double %calltmp
}

ready> c(3);
Parsed a top level expr

define double @1() {
entry:
  %calltmp = call double @c(double 3.000000e+00)
  ret double %calltmp
}

Executing function
Returned 25
```
