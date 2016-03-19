extern crate libc;
extern crate llvm_sys as llvm;

use std::collections::HashMap;

// use lib::llvm::{BuilderRef, ContextRef, ExecutionEngineRef, False, ModuleRef, PassManagerRef, RealULT, TypeRef, ValueRef, PrintMessageAction};
// use lib::llvm::llvm;

use std::char;
use std::io;
use std::io::stdio;
use std::str;
use std::vec;
use libc::{c_uint};

#[derive(Clone)]
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
      Char('+') => return "addtmp".with_c_str(|name| llvm::core::LLVMBuildFAdd(parser.builderRef, lhsValue, rhsValue, name)),
      Char('-') => return "subtmp".with_c_str(|name| llvm::core::LLVMBuildFSub(parser.builderRef, lhsValue, rhsValue, name)),
      Char('*') => return "multmp".with_c_str(|name| llvm::core::LLVMBuildFMul(parser.builderRef, lhsValue, rhsValue, name)),
      Char('<') => {

        let cmpValue = "cmptmp".with_c_str(|name| llvm::core::LLVMBuildFCmp(parser.builderRef, RealULT as c_uint, lhsValue, rhsValue, name));
        let ty = llvm::core::LLVMDoubleTypeInContext(parser.contextRef);
        return "booltmp".with_c_str(|name| llvm::core::LLVMBuildUIToFP(parser.builderRef, cmpValue, ty, name));
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
    let calleeF = self.callee.with_c_str(|name| llvm::core::LLVMGetOrInsertFunction(parser.moduleRef, name, funType));

    // TODO check arg size
    let mut argsV : Vec<ValueRef> = Vec::new();
    for arg in self.args.iter() {
      argsV.push(arg.codegen(parser));
    }

    return "calltmp".with_c_str(|name| llvm::core::LLVMBuildCall(parser.builderRef, calleeF, argsV.as_ptr(), argsV.len() as c_uint, name));
  }
}

impl PrototypeAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> ValueRef {
    let funType = parser.getDoubleFunType(self.argNames.len());
    let fun = self.name.with_c_str(|name| llvm::core::LLVMGetOrInsertFunction(parser.moduleRef, name, funType));
    if llvm::core::LLVMCountBasicBlocks(fun) != 0 {
      panic!("Redefinition of function");
    }
    let nArgs = llvm::core::LLVMCountParams(fun) as uint;
    if nArgs != 0 && nArgs != self.argNames.len() {
      panic!("Redefinition of function with different argument count");
    }

    for (i, argName) in self.argNames.iter().enumerate() {
      let llarg = llvm::core::LLVMGetParam(fun, i as c_uint);
      argName.with_c_str(|name| llvm::core::LLVMSetValueName(llarg, name));
      parser.namedValues.insert(argName.clone(), llarg);
    }

    return fun;
  }
}

impl FunctionAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> ValueRef {
    parser.namedValues.clear();

    let fun = self.proto.codegen(parser);
    let basicBlock = "entry".with_c_str(|name| llvm::core::LLVMAppendBasicBlockInContext(parser.contextRef, fun, name));
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
  tokenInput: Receiver<Token>,
  currentToken: Token,
  moduleRef: ModuleRef,
  builderRef: BuilderRef,
  contextRef: ContextRef,
  executionEngineRef: ExecutionEngineRef,
  functionPassManagerRef: PassManagerRef,
  namedValues: HashMap<String, ValueRef>
}

type ParseResult<T> = Result<T, String>;

impl Parser {
  fn new(tokenInput: Receiver<Token>) -> Parser {
    unsafe {
      if llvm::core::LLVMRustInitializeNativeTarget() != 0 {
        panic!("initializing native target");
      }
    }

    let llcx = llvm::core::LLVMContextCreate();
    let llmod = unsafe {
      "kaleidoscope".with_c_str(|name| {
        llvm::core::LLVMModuleCreateWithNameInContext(name, llcx)
      })
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
    let mut args: Vec<ExprAst> = Vec::new();
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
    let lhs: ExprAst = match self.parsePrimary() {
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

fn readTokens(tokenSender: Sender<Token>) -> fn() {
  return || {
    let mut reader = io::stdin();
    let mut lastChr = ' ';

    loop {
      while lastChr == ' ' || lastChr == '\r' || lastChr == '\n' || lastChr == '\t' {
        lastChr = match reader.read_char() {
          Ok(chr) => chr,
          Err(_) => break
        };
      }

      if char::is_alphabetic(lastChr) { // identifier [a-zA-Z][a-zA-Z0-9]*
        let mut identifierStr: Vec<char> = Vec::new();
        identifierStr.push(lastChr);

        loop {
          match reader.read_char() {
            Ok(chr) => {
              if char::is_alphabetic(chr) {
                identifierStr.push(chr);
              } else {
                lastChr = chr;
                break;
              }
            },
            Err(_) => {
              tokenSender.send(EndOfFile);
              return;
            }
          }
        }
        let identifier = str::from_chars(identifierStr.as_slice());
        if identifier == "def" {
          tokenSender.send(Token::Def);
        } else if identifier == "extern" {
          tokenSender.send(Token::Extern);
        } else {
          tokenSender.send(Token::Identifier(identifier));
        }
        continue;
      }

      if char::is_digit(lastChr, 10) || lastChr == '.' { // number: [0-9.]+
        let mut numStr: Vec<char> = Vec::new();
        numStr.push(lastChr);
        loop {
          match reader.read_char() {
            Ok(chr) => {
              if char::is_digit(chr, 10) || chr == '.' {
                numStr.push(chr);
              } else {
                lastChr = chr;
                break;
              }
            },
            Err(_) => {
              tokenSender.send(Token::EndOfFile);
              return;
            }
          }
        }
        tokenSender.send(Token::Number(match from_str::<f64>(str::from_chars(numStr.as_slice())) {
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
              tokenSender.send(Token::EndOfFile);
              return;
            }
          }
        }
        continue;
      }

      tokenSender.send(Token::Char(lastChr));
      // consume lastChr
      lastChr = ' ';
    }
  };
}

fn main() {
  let (tokenSender, tokenReceiver) = channel();

  spawn(readTokens(tokenSender));
  let mut parser = Parser::new(tokenReceiver);
  parser.run();
}
