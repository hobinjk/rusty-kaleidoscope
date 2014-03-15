Rusty Kaleidoscope
==================
An implementation of LLVM's wonderful Kaleidoscope tutorial in Rust. Uses
`librustc`'s LLVM binding because it exists.

Done
----
- Parsing
- Code generation
- Error detection

Todo
----
- Better error recovery
- JIT compilation/evaluation
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
```


