extern crate core;
extern crate libc;
extern crate llvm_sys as llvm;

use core::str::FromStr;

use std::collections::HashMap;

// use lib::llvm::llvm;
use llvm::prelude::{LLVMBuilderRef, LLVMContextRef, LLVMModuleRef, LLVMPassManagerRef, LLVMTypeRef, LLVMValueRef};
use llvm::execution_engine::{LLVMExecutionEngineRef, LLVMGenericValueToFloat, LLVMRunFunction, LLVMGenericValueRef};
use llvm::analysis::LLVMVerifyFunction;
use llvm::LLVMRealPredicate;

use std::char;
use std::ffi::CString;
use std::io::{self, Read, Write};
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
       (&Token::Def, &Token::Def) => true,
       (&Token::Extern, &Token::Extern) => true,
       (&Token::Identifier(ref val), &Token::Identifier(ref oVal)) => val == oVal,
       (&Token::Number(val), &Token::Number(oVal)) => val == oVal,
       (&Token::Char(val), &Token::Char(oVal)) => val == oVal,
       (&Token::EndOfFile, &Token::EndOfFile) => true,
       (_, _) => false,
     }
  }
}

impl Eq for Token {
}

trait ExprAst {
  unsafe fn codegen(&self, &mut Parser) -> LLVMValueRef;
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

fn cstr<'a>(input: &'a str) -> CString {
    return CString::new(input).unwrap()
}

impl ExprAst for NumberExprAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> LLVMValueRef {
    let ty = llvm::core::LLVMDoubleTypeInContext(parser.contextRef);
    return llvm::core::LLVMConstReal(ty, self.val);
  }
}

impl ExprAst for VariableExprAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> LLVMValueRef {
    return match parser.namedValues.get(&self.name) {
      Some(v) => *v,
      None => panic!("Unknown variable name {}", self.name)
    };
  }
}

impl ExprAst for BinaryExprAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> LLVMValueRef {
    let lhsValue = self.lhs.codegen(parser);
    let rhsValue = self.rhs.codegen(parser);
    match self.op {
      Token::Char('+') =>
        return llvm::core::LLVMBuildFAdd(parser.builderRef, lhsValue, rhsValue, cstr("addtmp").as_ptr()),
      Token::Char('-') =>
        return llvm::core::LLVMBuildFSub(parser.builderRef, lhsValue, rhsValue, cstr("subtmp").as_ptr()),
      Token::Char('*') =>
        return llvm::core::LLVMBuildFMul(parser.builderRef, lhsValue, rhsValue, cstr("multmp").as_ptr()),
      Token::Char('<') => {
        let cmpValue = llvm::core::LLVMBuildFCmp(parser.builderRef, LLVMRealULT, lhsValue, rhsValue, cstr("cmptmp").as_ptr());
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
  unsafe fn codegen(&self, parser: &mut Parser) -> LLVMValueRef {
    let funType : LLVMTypeRef = parser.getDoubleFunType(self.args.len());
    let calleeF = llvm::core::LLVMAddFunction(parser.moduleRef, cstr(&self.callee).as_ptr(), funType);

    // TODO check arg size
    let mut argsV : Vec<LLVMValueRef> = Vec::new();
    for arg in self.args.iter() {
      argsV.push(arg.codegen(parser));
    }

    return llvm::core::LLVMBuildCall(parser.builderRef, calleeF, argsV.as_mut_ptr(), argsV.len() as c_uint, cstr("calltmp").as_ptr());
  }
}

impl PrototypeAst {
  unsafe fn codegen(&self, parser: &mut Parser) -> LLVMValueRef {
    let funType = parser.getDoubleFunType(self.argNames.len());
    let fun = llvm::core::LLVMAddFunction(parser.moduleRef, cstr(&self.name).as_ptr(), funType);
    if llvm::core::LLVMCountBasicBlocks(fun) != 0 {
      panic!("Redefinition of function");
    }
    let nArgs = llvm::core::LLVMCountParams(fun) as usize;
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
  unsafe fn codegen(&self, parser: &mut Parser) -> LLVMValueRef {
    parser.namedValues.clear();

    let fun = self.proto.codegen(parser);
    let basicBlock = llvm::core::LLVMAppendBasicBlockInContext(parser.contextRef, fun, cstr("entry").as_ptr());
    llvm::core::LLVMPositionBuilderAtEnd(parser.builderRef, basicBlock);
    let body = self.body.codegen(parser);
    llvm::core::LLVMBuildRet(parser.builderRef, body);

    if llvm::core::LLVMVerifyFunction(fun, LLVMPrintMessageAction) != 0 {
      println!("Function verify failed");
    }

    llvm::core::LLVMRunFunctionPassManager(parser.functionPassManagerRef, fun);

    return fun;
  }
}


struct Parser {
  tokens: Vec<Token>,
  tokenIndex: usize,
  currentToken: Token,
  moduleRef: LLVMModuleRef,
  builderRef: LLVMBuilderRef,
  contextRef: LLVMContextRef,
  executionEngineRef: LLVMExecutionEngineRef,
  functionPassManagerRef: LLVMPassManagerRef,
  namedValues: HashMap<String, LLVMValueRef>
}

type ParseResult<T> = Result<T, &'static str>;

impl Parser {
  fn new(tokens: Vec<Token>) -> Parser {
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
      let llee: LLVMExecutionEngineRef = 0 as LLVMExecutionEngineRef;
      let err: *mut char = 0 as *mut char;
      llvm::core::LLVMCreateExecutionEngineForModule(&llee, llmod, &err);
      llee
    };
    return Parser {
      tokens: tokens,
      tokenIndex: 0,
      currentToken: Token::Char(' '),
      moduleRef: llmod,
      builderRef: llbuilder,
      contextRef: llcx,
      executionEngineRef: llee,
      functionPassManagerRef: llfpm,
      namedValues: HashMap::new()
    };
  }

  unsafe fn getDoubleFunType(&mut self, argc: usize) -> LLVMTypeRef {
    let ty = llvm::core::LLVMDoubleTypeInContext(self.contextRef);
    let doubles: Vec<LLVMTypeRef> = (0..argc).map(|_| ty).collect();
    return llvm::core::LLVMFunctionType(ty, doubles.as_mut_ptr(), argc as c_uint, 0);
  }

  fn getNextToken(&mut self) {
    self.currentToken = self.tokens[self.tokenIndex];
    self.tokenIndex += 1;
  }

  fn parseNumberExpr(&mut self) -> ParseResult<Box<ExprAst>> {
    let val = match self.currentToken {
      Token::Number(val) => val,
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
      Token::Char(')') => {},
      _ => return Err("expected ')'")
    }
    self.getNextToken();
    return Ok(expr);
  }

  fn parseIdentifierExpr(&mut self) -> ParseResult<Box<ExprAst>> {
    let idName = match self.currentToken {
      Token::Identifier(ref name) => name.clone(),
      _ => return Err("token not an identifier")
    };

    self.getNextToken();

    match self.currentToken {
      Token::Char('(') => {},
      _ => return Ok(Box::new(VariableExprAst{name: idName}))
    }

    self.getNextToken();
    let mut args: Vec<Box<ExprAst>> = Vec::new();
    if self.currentToken != Token::Char(')') {
      loop {
        let arg = self.parseExpression();
        match arg {
          Ok(arg) => args.push(arg),
          err => return err
        }

        if self.currentToken == Token::Char(')') {
          break;
        }

        if self.currentToken != Token::Char(',') {
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
      Token::Identifier(_) => return self.parseIdentifierExpr(),
      Token::Number(_) => return self.parseNumberExpr(),
      Token::Char('(') => return self.parseParenExpr(),
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

  fn parseBinOpRhs(&mut self, exprPrec: i32, startLhs: Box<ExprAst>) -> ParseResult<Box<ExprAst>> {
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
      Token::Identifier(ref name) => name.clone(),
      _ => return Err("Expected function name in prototype")
    };

    self.getNextToken();
    if self.currentToken != Token::Char('(') {
      println!("had a {:?}", self.currentToken);
      return Err("Expected '(' in prototype");
    }

    let mut argNames: Vec<String> = Vec::new();
    loop {
      self.getNextToken();
      match self.currentToken {
        Token::Identifier(ref name) => argNames.push(name.clone()),
        _ => break
      }
    }
    if self.currentToken != Token::Char(')') {
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

  fn getTokenPrecedence(&mut self) -> i32 {
    match self.currentToken {
      Token::Char(t) => match t {
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
    io::stdout().flush();
    self.getNextToken();

    loop {
      match self.currentToken {
        Token::Def => self.handleDefinition(),
        Token::Extern => self.handleExtern(),
        Token::Char(';') => {
          self.getNextToken();
          continue;
        },
        _ => self.handleTopLevelExpression()
      }

      print!("ready> ");
      io::stdout().flush();
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
          let mut argsV: Vec<LLVMGenericValueRef> = Vec::new();
          println!("Executing function");
          let retValue = LLVMRunFunction(self.executionEngineRef, tleFun, argsV.len() as c_uint, argsV.as_mut_ptr());
          let doubleTy = llvm::core::LLVMDoubleTypeInContext(self.contextRef);
          let fl = LLVMGenericValueToFloat(doubleTy, retValue);
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
      tokens.push(Token::Number(match f64::from_str(&numStr) {
        Ok(val) => val,
        Err(_) => {
          println!("Malformed number");
          continue;
        }
      }));
      continue;
    }

    if lastChr == '#' {
      loop {
        match chars.next() {
          Some(chr) => {
            if chr == '\r' || chr == '\n' {
              lastChr = ' ';
              break;
            }
          },
          None => {
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
  let mut parser = Parser::new(tokens);
  parser.run();
}
