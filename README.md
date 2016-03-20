Rusty Kaleidoscope
==================
An implementation of LLVM's wonderful Kaleidoscope tutorial in Rust. Uses
`llvm-sys`'s LLVM binding.

Compilation
-----------
Moving to Cargo made this much easier! Just run `cargo build`!

Done
----
- Parsing
- Code generation
- Error detection
- JIT compilation/evaluation

Todo
----
- Top Level Expr's don't work
- Exercise 4 onward

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
