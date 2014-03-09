use std::char;
use std::io;
use std::io::stdio;
use std::str;

#[deriving(Clone)]
enum Token {
  Def,
  Extern,
  Identifier(~str),
  Number(f64),
  Char(char),
  EndOfFile
}

impl Eq for Token {
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

trait ExprAst {
}

struct NumberExprAst {
  val: f64
}

struct VariableExprAst {
  name: ~str
}

struct BinaryExprAst {
  op: Token,
  lhs: ~ExprAst,
  rhs: ~ExprAst,
}

struct CallExprAst {
  callee: ~str,
  args: ~[~ExprAst]
}

struct PrototypeAst {
  name: ~str,
  argNames: ~[~str]
}

struct FunctionAst {
  proto: ~PrototypeAst,
  body: ~ExprAst
}

impl ExprAst for NumberExprAst {
}
impl ExprAst for VariableExprAst {
}
impl ExprAst for BinaryExprAst {
}
impl ExprAst for CallExprAst {
}

struct Parser {
  tokenInput: Port<Token>,
  currentToken: Token
}

type ParseResult<T> = Result<T, ~str>;

impl Parser {
  fn getNextToken(&mut self) {
    self.currentToken = self.tokenInput.recv();
  }

  fn parseNumberExpr(&mut self) -> ParseResult<~ExprAst> {
    let val = match self.currentToken {
      Number(val) => val,
      _ => return Err(~"token not a number")
    };

    let expr = ~NumberExprAst{val: val};
    self.getNextToken();
    return Ok(expr as ~ExprAst);
  }

  fn parseParenExpr(&mut self) -> ParseResult<~ExprAst> {
    self.getNextToken();
    let expr = match self.parseExpression() {
      Ok(expr) => expr,
      err => return err
    };

    match self.currentToken {
      Char(')') => {},
      _ => return Err(~"expected ')'")
    }
    self.getNextToken();
    return Ok(expr);
  }

  fn parseIdentifierExpr(&mut self) -> ParseResult<~ExprAst> {
    let idName = match self.currentToken {
      Identifier(ref name) => name.clone(),
      _ => return Err(~"token not an identifier")
    };

    self.getNextToken();

    match self.currentToken {
      Char('(') => {},
      _ => return Ok(~VariableExprAst{name: idName} as ~ExprAst)
    }

    self.getNextToken();
    let mut args: ~[~ExprAst] = ~[];
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
          return Err(~"Expected ')' or ',' in argument list");
        }

        self.getNextToken();
      }
    }

    self.getNextToken();

    return Ok(~CallExprAst {callee: idName, args: args} as ~ExprAst);
  }

  fn parsePrimary(&mut self) -> ParseResult<~ExprAst> {
    match self.currentToken {
      Identifier(_) => return self.parseIdentifierExpr(),
      Number(_) => return self.parseNumberExpr(),
      Char('(') => return self.parseParenExpr(),
      _ => return Err(~"unknown token when expecting an expression")
    }
  }

  fn parseExpression(&mut self) -> ParseResult<~ExprAst> {
    let lhs: ~ExprAst = match self.parsePrimary() {
      Ok(lhs) => lhs,
      err => return err
    };
    return self.parseBinOpRhs(0, lhs);
  }

  fn parseBinOpRhs(&mut self, exprPrec: int, startLhs: ~ExprAst) -> ParseResult<~ExprAst> {
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
      lhs = ~BinaryExprAst {op: binOp, lhs: lhs, rhs: rhs} as ~ExprAst;
    }
  }

  fn parsePrototype(&mut self) -> ParseResult<~PrototypeAst> { // possibly need sep. of Prototype and Expr
    let fnName: ~str = match self.currentToken {
      Identifier(ref name) => name.clone(),
      _ => return Err(~"Expected function name in prototype")
    };

    self.getNextToken();
    if self.currentToken != Char('(') {
      println!("had a {:?}", self.currentToken);
      return Err(~"Expected '(' in prototype");
    }

    let mut argNames: ~[~str] = ~[];
    loop {
      self.getNextToken();
      match self.currentToken {
        Identifier(ref name) => argNames.push(name.clone()),
        _ => break
      }
    }
    if self.currentToken != Char(')') {
      return Err(~"Expected ')' in prototype");
    }

    self.getNextToken();

    return Ok(~PrototypeAst {name: fnName, argNames: argNames});
  }

  fn parseDefinition(&mut self) -> ParseResult<~FunctionAst> {
    self.getNextToken();
    let proto = match self.parsePrototype() {
      Ok(proto) => proto,
      Err(err) => return Err(err)
    };
    let expr = match self.parseExpression() {
      Ok(expr) => expr,
      Err(err) => return Err(err)
    };
    return Ok(~FunctionAst{proto: proto, body: expr});
  }

  fn parseExtern(&mut self) -> ParseResult<~PrototypeAst> {
    self.getNextToken(); // consume "expr"
    return self.parsePrototype();
  }

  fn parseTopLevelExpr(&mut self) -> ParseResult<~FunctionAst> {
    let expr = match self.parseExpression() {
      Ok(expr) => expr,
      Err(err) => return Err(err)
    };

    let proto = ~PrototypeAst {name: ~"", argNames: ~[]};
    return Ok(~FunctionAst{proto: proto, body: expr});
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
      Ok(def) => println!("Parsed a function definition"),
      Err(why) => {
        println!("Error: {}", why);
        self.getNextToken();
      }
    }
  }

  fn handleExtern(&mut self) {
    let ext = self.parseExtern();
    match ext {
      Ok(ext) => println!("Parsed an extern"),
      Err(why) => {
        println!("Error parsing extern: {}", why);
        self.getNextToken();
      }
    }
  }

  fn handleTopLevelExpression(&mut self) {
    let tle = self.parseTopLevelExpr();
    match tle {
      Ok(tle) => println!("Parsed a top level expr"),
      Err(why) => {
        println!("Error parsing tle: {}", why);
        self.getNextToken();
      }
    }
  }
}

fn readTokens(tokenChan: Chan<Token>) -> proc() {
  return proc() {
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
        let mut identifierStr : ~[char] = ~[lastChr];
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
              tokenChan.send(EndOfFile);
              return;
            }
          }
        }
        match str::from_chars(identifierStr) {
          ~"def" => {
            tokenChan.send(Def);
            continue;
          },
          ~"extern" => {
            tokenChan.send(Extern);
            continue;
          },
          id => {
            tokenChan.send(Identifier(id));
            continue;
          }
        }
      }

      if char::is_digit(lastChr) || lastChr == '.' { // number: [0-9.]+
        let mut numStr = ~[lastChr];
        loop {
          match reader.read_char() {
            Ok(chr) => {
              if char::is_digit(chr) || chr == '.' {
                numStr.push(chr);
              } else {
                lastChr = chr;
                break;
              }
            },
            Err(_) => {
              tokenChan.send(EndOfFile);
              return;
            }
          }
        }
        tokenChan.send(Number(match from_str::<f64>(str::from_chars(numStr)) {
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
              tokenChan.send(EndOfFile);
              return;
            }
          }
        }
        continue;
      }

      tokenChan.send(Char(lastChr));
      // consume lastChr
      lastChr = ' ';
    }
    tokenChan.send(EndOfFile);
  };
}

fn main() {
  let (tokenPort, tokenChan) = Chan::new();

  spawn(readTokens(tokenChan));
  let mut parser = Parser {tokenInput: tokenPort, currentToken: Char(' ')};
  parser.run();
}
