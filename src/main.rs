extern crate libc;
extern crate llvm_sys as llvm;

use std::collections::HashMap;

// use lib::llvm::{BuilderRef, ContextRef, ExecutionEngineRef, False, ModuleRef, PassManagerRef, RealULT, TypeRef, ValueRef, PrintMessageAction};
// use lib::llvm::llvm;

use std::char;
use std::ffi;
use std::io::{self, Read};
use std::str;
use std::vec;
use libc::{c_uint};

#[derive(Clone,Debug)]
enum Token {
  Def,
  Extern,
  Identifier(String),
  Number(f64),
  Char(char),
  EndOfFile
}
impl PartialEq for Token {
  fn eq(&self, other: &Token) -> bool {
     match (self, other) {
       (&Def, &Def) => true,
       (&Extern, &Extern) => true,
       (&Identifier(ref val), &Identifier(ref oVal)) => val == oVal,
       (&Number(val), &Number(oVal)) => val == oVal,
       (&Char(val), &Char(oVal)) => val == oVal,
       (&EndOfFile, &EndOfFile) => true,
       (_, _) => false,
     }
  }
}

impl Eq for Token {
}

trait ExprAst {
  unsafe fn codegen(&self, &mut Parser) -> ValueRef;
}

struct NumberExprAst {
  val: f64
}

struct VariableExprAst {
  name: String
}

struct BinaryExprAst {
  op: Token,
  lhs: Box<ExprAst>,
  rhs: Box<ExprAst>,
}

struct CallExprAst {
  callee: String,
  args: Vec<Box<ExprAst>>
}

struct PrototypeAst {
  name: String,
  argNames: Vec<String>
}

struct FunctionAst {
  proto: Box<PrototypeAst>,
  body: Box<ExprAst>
}

fn cstr<'a>(input: &'a str) -> &'a CString {
    return CString::new(input).unwrap()
}

impl ExprAst for NumberExprAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> ValueRef {
    let ty = llvm::core::LLVMDoubleTypeInContext(parser.contextRef);
    return llvm::core::LLVMConstReal(ty, self.val);
  }
}

impl ExprAst for VariableExprAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> ValueRef {
    return match parser.namedValues.find_copy(&self.name) {
      Some(v) => v,
      None => panic!("Unknown variable name {}", self.name)
    };
  }
}

impl ExprAst for BinaryExprAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> ValueRef {
    let lhsValue = self.lhs.codegen(parser);
    let rhsValue = self.rhs.codegen(parser);
    match self.op {
      Char('+') =>
        return llvm::core::LLVMBuildFAdd(parser.builderRef, lhsValue, rhsValue, cstr("addtmp").as_ptr()),
      Char('-') =>
        return llvm::core::LLVMBuildFSub(parser.builderRef, lhsValue, rhsValue, cstr("subtmp").as_ptr()),
      Char('*') =>
        return llvm::core::LLVMBuildFMul(parser.builderRef, lhsValue, rhsValue, cstr("multmp").as_ptr()),
      Char('<') => {
        let cmpValue = llvm::core::LLVMBuildFCmp(parser.builderRef, RealULT as c_uint, lhsValue, rhsValue, cstr("cmptmp").as_ptr());
        let ty = llvm::core::LLVMDoubleTypeInContext(parser.contextRef);
        return llvm::core::LLVMBuildUIToFP(parser.builderRef, cmpValue, ty, cstr("booltmp").as_ptr());
      }
      _ => {
        panic!("llvm code gen failed, invalid binary operation");
      }
    }

  }
}

impl ExprAst for CallExprAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> ValueRef {
    let funType : TypeRef = parser.getDoubleFunType(self.args.len());
    let calleeF = llvm::core::LLVMGetOrInsertFunction(parser.moduleRef, cstr(self.callee).as_ptr(), funType);

    // TODO check arg size
    let mut argsV : Vec<ValueRef> = Vec::new();
    for arg in self.args.iter() {
      argsV.push(arg.codegen(parser));
    }

    return llvm::core::LLVMBuildCall(parser.builderRef, calleeF, argsV.as_ptr(), argsV.len() as c_uint, cstr("calltmp").as_ptr());
  }
}

impl PrototypeAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> ValueRef {
    let funType = parser.getDoubleFunType(self.argNames.len());
    let fun = llvm::core::LLVMGetOrInsertFunction(parser.moduleRef, cstr(self.name).as_ptr(), funType);
    if llvm::core::LLVMCountBasicBlocks(fun) != 0 {
      panic!("Redefinition of function");
    }
    let nArgs = llvm::core::LLVMCountParams(fun) as uint;
    if nArgs != 0 && nArgs != self.argNames.len() {
      panic!("Redefinition of function with different argument count");
    }

    for (i, argName) in self.argNames.iter().enumerate() {
      let llarg = llvm::core::LLVMGetParam(fun, i as c_uint);
      llvm::core::LLVMSetValueName(llarg, cstr(argName).as_ptr());
      parser.namedValues.insert(argName.clone(), llarg);
    }

    return fun;
  }
}

impl FunctionAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> ValueRef {
    parser.namedValues.clear();

    let fun = self.proto.codegen(parser);
    let basicBlock = llvm::core::LLVMAppendBasicBlockInContext(parser.contextRef, fun, cstr("entry").as_ptr());
    llvm::core::LLVMPositionBuilderAtEnd(parser.builderRef, basicBlock);
    let body = self.body.codegen(parser);
    llvm::core::LLVMBuildRet(parser.builderRef, body);

    if llvm::core::LLVMVerifyFunction(fun, PrintMessageAction as c_uint) != 0 {
      println!("Function verify failed");
    }

    llvm::core::LLVMRunFunctionPassManager(parser.functionPassManagerRef, fun);

    return fun;
  }
}


struct Parser {
  tokenInput: Iter<Token>,
  currentToken: Token,
  moduleRef: ModuleRef,
  builderRef: BuilderRef,
  contextRef: ContextRef,
  executionEngineRef: ExecutionEngineRef,
  functionPassManagerRef: PassManagerRef,
  namedValues: HashMap<String, ValueRef>
}

type ParseResult<T> = Result<T, &'static str>;

impl Parser {
  fn new(tokens: Iter<Token>) -> Parser {
    unsafe {
      if llvm::core::LLVMRustInitializeNativeTarget() != 0 {
        panic!("initializing native target");
      }
    }

    let llcx = llvm::core::LLVMContextCreate();
    let llmod = unsafe {
      llvm::core::LLVMModuleCreateWithNameInContext(cstr("kaleidoscope").as_ptr(), llcx)
    };
    let llfpm = unsafe {
      llvm::core::LLVMCreateFunctionPassManagerForModule(llmod)
    };
    unsafe {
      llvm::core::LLVMAddBasicAliasAnalysisPass(llfpm);
      llvm::core::LLVMAddInstructionCombiningPass(llfpm);
      llvm::core::LLVMAddReassociatePass(llfpm);
      llvm::core::LLVMAddGVNPass(llfpm);
      llvm::core::LLVMAddCFGSimplificationPass(llfpm);

      llvm::core::LLVMInitializeFunctionPassManager(llfpm);
    }

    let llbuilder = unsafe {
      llvm::core::LLVMCreateBuilderInContext(llcx)
    };

    let llee = unsafe {
      // initialize vars to NULL
      let llee: ExecutionEngineRef = 0 as ExecutionEngineRef;
      let err: *mut char = 0 as *mut char;
      llvm::core::LLVMCreateExecutionEngineForModule(&llee, llmod, &err);
      llee
    };
    return Parser {
      tokenInput: tokenInput,
      currentToken: Char(' '),
      moduleRef: llmod,
      builderRef: llbuilder,
      contextRef: llcx,
      executionEngineRef: llee,
      functionPassManagerRef: llfpm,
      namedValues: HashMap::new()
    };
  }

  unsafe fn getDoubleFunType(&mut self, argc: uint) -> TypeRef {
    let ty = llvm::core::LLVMDoubleTypeInContext(self.contextRef);
    let doubles: Vec<TypeRef> = Vec::from_fn(argc, |_| ty);
    return llvm::core::LLVMFunctionType(ty, doubles.as_ptr(), argc as c_uint, False);
  }

  fn getNextToken(&mut self) {
    self.currentToken = self.tokenInput.recv();
  }

  fn parseNumberExpr(&mut self) -> ParseResult<Box<ExprAst>> {
    let val = match self.currentToken {
      Number(val) => val,
      _ => return Err("token not a number")
    };

    let expr = Box::new(NumberExprAst{val: val});
    self.getNextToken();
    return Ok(expr);
  }

  fn parseParenExpr(&mut self) -> ParseResult<Box<ExprAst>> {
    self.getNextToken();
    let expr = match self.parseExpression() {
      Ok(expr) => expr,
      err => return err
    };

    match self.currentToken {
      Char(')') => {},
      _ => return Err("expected ')'")
    }
    self.getNextToken();
    return Ok(expr);
  }

  fn parseIdentifierExpr(&mut self) -> ParseResult<Box<ExprAst>> {
    let idName = match self.currentToken {
      Identifier(ref name) => name.clone(),
      _ => return Err("token not an identifier")
    };

    self.getNextToken();

    match self.currentToken {
      Char('(') => {},
      _ => return Ok(Box::new(VariableExprAst{name: idName}))
    }

    self.getNextToken();
    let mut args: Vec<Box<ExprAst>> = Vec::new();
    if self.currentToken != Char(')') {
      loop {
        let arg = self.parseExpression();
        match arg {
          Ok(arg) => args.push(arg),
          err => return err
        }

        if self.currentToken == Char(')') {
          break;
        }

        if self.currentToken != Char(',') {
          return Err("Expected ')' or ',' in argument list");
        }

        self.getNextToken();
      }
    }

    self.getNextToken();

    return Ok(Box::new(CallExprAst {callee: idName, args: args}));
  }

  fn parsePrimary(&mut self) -> ParseResult<Box<ExprAst>> {
    match self.currentToken {
      Identifier(_) => return self.parseIdentifierExpr(),
      Number(_) => return self.parseNumberExpr(),
      Char('(') => return self.parseParenExpr(),
      _ => return Err("unknown token when expecting an expression")
    }
  }

  fn parseExpression(&mut self) -> ParseResult<Box<ExprAst>> {
    let lhs: Box<ExprAst> = match self.parsePrimary() {
      Ok(lhs) => lhs,
      err => return err
    };
    return self.parseBinOpRhs(0, lhs);
  }

  fn parseBinOpRhs(&mut self, exprPrec: int, startLhs: Box<ExprAst>) -> ParseResult<Box<ExprAst>> {
    let mut lhs = startLhs;
    loop {
      let tokenPrec = self.getTokenPrecedence();
      if tokenPrec < exprPrec {
        return Ok(lhs);
      }

      let binOp = self.currentToken.clone();
      self.getNextToken();

      let mut rhs = match self.parsePrimary() {
        Ok(rhs) => rhs,
        err => return err
      };

      let nextPrec = self.getTokenPrecedence();

      if tokenPrec < nextPrec {
        rhs = match self.parseBinOpRhs(tokenPrec+1, rhs) {
          Ok(rhs) => rhs,
          err => return err
        };
      }
      lhs = Box::new(BinaryExprAst {op: binOp, lhs: lhs, rhs: rhs});
    }
  }

  fn parsePrototype(&mut self) -> ParseResult<Box<PrototypeAst>> { // possibly need sep. of Prototype and Expr
    let fnName: String = match self.currentToken {
      Identifier(ref name) => name.clone(),
      _ => return Err("Expected function name in prototype")
    };

    self.getNextToken();
    if self.currentToken != Char('(') {
      println!("had a {:?}", self.currentToken);
      return Err("Expected '(' in prototype");
    }

    let mut argNames: Vec<String> = Vec::new();
    loop {
      self.getNextToken();
      match self.currentToken {
        Identifier(ref name) => argNames.push(name.clone()),
        _ => break
      }
    }
    if self.currentToken != Char(')') {
      return Err("Expected ')' in prototype");
    }

    self.getNextToken();

    return Ok(Box::new(PrototypeAst {name: fnName, argNames: argNames}));
  }

  fn parseDefinition(&mut self) -> ParseResult<Box<FunctionAst>> {
    self.getNextToken();
    let proto = match self.parsePrototype() {
      Ok(proto) => proto,
      Err(err) => return Err(err)
    };
    let expr = match self.parseExpression() {
      Ok(expr) => expr,
      Err(err) => return Err(err)
    };
    return Ok(Box::new(FunctionAst{proto: proto, body: expr}));
  }

  fn parseExtern(&mut self) -> ParseResult<Box<PrototypeAst>> {
    self.getNextToken(); // consume "expr"
    return self.parsePrototype();
  }

  fn parseTopLevelExpr(&mut self) -> ParseResult<Box<FunctionAst>> {
    let expr = match self.parseExpression() {
      Ok(expr) => expr,
      Err(err) => return Err(err)
    };

    let proto = Box::new(PrototypeAst {name: "".to_string(), argNames: Vec::new()});
    return Ok(Box::new(FunctionAst{proto: proto, body: expr}));
  }

  fn getTokenPrecedence(&mut self) -> int {
    match self.currentToken {
      Char(t) => match t {
        '<' => return 10,
        '+' => return 20,
        '-' => return 20,
        '*' => return 40,
        '/' => return 40,
        _   => return -1
      },
      _ => return -1
    }
  }

  fn run(&mut self) {
    print!("ready> ");
    stdio::flush();
    self.getNextToken();

    loop {
      match self.currentToken {
        Def => self.handleDefinition(),
        Extern => self.handleExtern(),
        Char(';') => {
          self.getNextToken();
          continue;
        },
        _ => self.handleTopLevelExpression()
      }

      print!("ready> ");
      stdio::flush();
    }
  }

  fn handleDefinition(&mut self) {
    let def = self.parseDefinition();
    match def {
      Ok(def) => {
        println!("Parsed a function definition");
        unsafe {
          let fun = def.codegen(self);
          llvm::core::LLVMDumpValue(fun);
        }
      }
      Err(why) => {
        println!("Error: {}", why);
        self.getNextToken();
      }
    }
  }

  fn handleExtern(&mut self) {
    let ext = self.parseExtern();
    match ext {
      Ok(ext) => {
        println!("Parsed an extern");
        unsafe {
          let extLL = ext.codegen(self);
          llvm::core::LLVMDumpValue(extLL);
        }
      },
      Err(why) => {
        println!("Error parsing extern: {}", why);
        self.getNextToken();
      }
    }
  }

  fn handleTopLevelExpression(&mut self) {
    let tle = self.parseTopLevelExpr();
    match tle {
      Ok(tle) => {
        println!("Parsed a top level expr");
        unsafe {
          let tleFun = tle.codegen(self);
          llvm::core::LLVMDumpValue(tleFun);
          // we have a 0 arg function, call it using the executionEngineRef
          let argsV: Vec<ValueRef> = Vec::new();
          println!("Executing function");
          let retValue = llvm::core::LLVMRunFunction(self.executionEngineRef, tleFun, argsV.len() as c_uint, argsV.as_ptr());
          let doubleTy = llvm::core::LLVMDoubleTypeInContext(self.contextRef);
          let fl = llvm::core::LLVMGenericValueToFloat(doubleTy, retValue);
          println!("Returned {}", fl);
        }
      },
      Err(why) => {
        println!("Error parsing tle: {}", why);
        self.getNextToken();
      }
    }
  }
}

fn readTokens() -> Vec<Token> {
  let mut tokens = Vec::new();
  let mut buffer = String::new();
  match io::stdin().read_to_string(&mut buffer) {
      Ok(_) => {}
      Err(_) => return tokens
  }

  let mut chars = buffer.chars();
  let mut lastChr = ' ';

  loop {
    while lastChr == ' ' || lastChr == '\r' || lastChr == '\n' || lastChr == '\t' {
      lastChr = match chars.next() {
        Some(chr) => chr,
        None => break
      };
    }

    if lastChr.is_alphabetic() { // identifier [a-zA-Z][a-zA-Z0-9]*
      let mut identifier = String::new();
      identifier.push(lastChr);

      loop {
        match chars.next() {
          Some(chr) => {
            if chr.is_alphabetic() {
              identifier.push(chr);
            } else {
              lastChr = chr;
              break;
            }
          },
          None => {
            tokens.push(Token::EndOfFile);
            return tokens;
          }
        }
      }
      if identifier == "def" {
        tokens.push(Token::Def);
      } else if identifier == "extern" {
        tokens.push(Token::Extern);
      } else {
        tokens.push(Token::Identifier(identifier));
      }
      continue;
    }

    if char::is_digit(lastChr, 10) || lastChr == '.' { // number: [0-9.]+
      let mut numStr = String::new();
      numStr.push(lastChr);
      loop {
        match chars.next() {
          Some(chr) => {
            if char::is_digit(chr, 10) || chr == '.' {
              numStr.push(chr);
            } else {
              lastChr = chr;
              break;
            }
          },
          None => {
            tokens.push(Token::EndOfFile);
            return tokens;
          }
        }
      }
      tokens.push(Token::Number(match from_str::<f64>(&numStr) {
        Some(val) => val,
        None => {
          println!("Malformed number");
          continue;
        }
      }));
      continue;
    }

    if lastChr == '#' {
      loop {
        match reader.read_char() {
          Ok(chr) => {
            if chr == '\r' || chr == '\n' {
              lastChr = ' ';
              break;
            }
          },
          Err(_) => {
            tokens.push(Token::EndOfFile);
            return tokens;
          }
        }
      }
      continue;
    }

    tokens.push(Token::Char(lastChr));
    // consume lastChr
    lastChr = ' ';
  }

  return tokens;
}

fn main() {
  let tokens = readTokens();
  let mut parser = Parser::new(tokens.iter());
  parser.run();
}
