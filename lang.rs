use std::char;
use std::io;
use std::io::stdio;

enum Token {
  Def,
  Extern,
  Identifier(~str),
  Number(f64),
  Char(char),
  EndOfFile
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

struct Parser {
  tokenInput: Chan<Token>,
  currentToken: Token
}

type ParseResult = Result<~ExprAst, ~str>;

impl Parser {
  fn getNextToken(&mut self) {
    self.currentToken = self.tokenInput.recv();
  }

  fn parseNumberExpr(&mut self) -> ParseResult {
    let val = match self.currentToken {
      Number(val) => val,
      _ => return Err(~"token not a number")
    };

    let expr = ~NumberExprAst{val: val};
    self.getNextToken();
    return Ok(expr);
  }

  fn parseParenExpr(&mut self) -> ParseResult {
    self.getNextToken();
    let expr = self.parseExpression();
    match self.currentToken {
      Char(')') => {},
      _ => return Err(~"expected ')'")
    }
    self.getNextToken();
    return Ok(expr);
  }

  fn parseIdentifierExpr(&mut self) -> ParseResult {
    let idName = match self.currentToken {
      Identifier(name) => name,
      _ => return Err(~"token not an identifier")
    };

    self.getNextToken();

    match self.currentToken {
      Char('(') => {},
      _ => return Ok(VariableExprAst{name: idName})
    }

    self.getNextToken();
    let mut args: ~[~ExprAst] = ~[];
    if self.currentToken != Char(')') {
      loop {
        let arg = self.parseExpression();
        args.push(arg);

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

    return Ok(CallExprAst {name: idName, args: args});
  }

  fn parsePrimary(&mut self) -> ParseResult {
    match self.currentToken {
      Identifier(_) => return Ok(self.parseIdentifierExpr()),
      Number(_) => return Ok(self.parseNumberExpr()),
      '(' => return Ok(self.parseParenExpr()),
      what => return Err(format!("unknown token {:?} when expecting an expression", what))
    }
  }

  fn parseExpression(&mut self) -> ParseResult {
    let lhs: ~ExprAst = match self.parsePrimary() {
      Ok(lhs) => lhs,
      err => return err
    };
    return self.parseBinOpRhs(0, lhs);
  }

  fn parseBinOpRhs(&mut self, exprPrec: int, startLhs: ~ExprAst) -> ParseResult {
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
        rhs = self.parseBinOpRhs(tokenPrec+1, rhs);
      }
      lhs = ~BinaryExprAst {binOp: binOp, lhs: lhs, rhs: rhs};
    }
  }

  fn parsePrototype(&mut self) -> ParseResult { // possibly need sep. of Prototype and Expr
    let fnName: ~str = match self.currentToken {
      Identifier(name) => name,
      _ => return Err(~"Expected function name in prototype")
    };

    self.getNextToken();
    if self.currentToken != Char('(') {
      return Err(~"Expected '(' in prototype");
    }

    let mut argNames: ~[~str] = ~[];
    loop {
      self.getNextToken();
      match self.currentToken {
        Identifier(name) => argNames.push(name),
        _ => break
      }
    }
    if self.currentToken != Char(')') {
      return Err(~"Expected ')' in prototype");
    }

    self.getNextToken();

    return Ok(~PrototypeAst {name: fnName, argNames: argNames});
  }

  fn parseDefinition(&mut self) -> ParseResult {
    self.getNextToken();
    let proto = self.parsePrototype().unwrap();
    let expr = self.parseExpression().unwrap();
    return FunctionAst{proto: proto, body: expr};
  }

  fn parseExtern(&mut self) -> ParseResult {
    self.getNextToken(); // consume "expr"
    return self.parsePrototype();
  }

  fn parseTopLevelExpr(&mut self) -> ParseResult {
    let expr = match self.parseExpression() {
      Ok(expr) => expr,
      err => return err
    };

    let proto = ~PrototypeAst {name: "", argNames: ~[]};
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
      print!("ready> ");
      stdio::flush();
      match self.currentToken {
        Def => self.handleDefinition(),
        Extern => self.handleExtern(),
        Char(';') => self.getNextToken(),
        _ => self.handleTopLevelExpression()
      }
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

fn readTokens(tokenChan: Chan<Token>) {
  return proc() {
    let mut reader = io::stdin();
    let mut lastChar = ' ';

    for chr in reader.chars() {
      if chr == ' ' {
        continue
      }
      if char::is_alphabetic(chr) { // identifier [a-zA-Z][a-zA-Z0-9]*
        let mut identifierStr : ~str = ~"";
        identifierStr += chr;
        loop {
          match reader.read_char() {
            Some(chr) => {
              if char::is_alphabetic(chr) {
                identifierStr += chr;
              } else {
                break;
              }
            },
            None => return EndOfFile
          }
        }
        match identifierStr {
          ~"def" => return Def,
          ~"extern" => return Extern,
          id => return Identifier(id)
        }
      }

      if char::is_digit(chr) || chr == '.' { // number: [0-9.]+
        let mut numStr = ~"";
        numStr += chr;
        loop {
          match reader.read_char() {
            Some(chr) => {
              if char::is_digit(chr) || chr == '.' {
                numStr += chr;
              } else {
                break;
              }
            },
            None => return EndOfFile
          }
        }
        return Number(from_str::<f64>(numStr));
      }

      if chr == '#' {
        loop {
          match reader.read_char() {
            Some(chr) => {
              if chr == '\r' || chr == '\n' {
                break;
              }
            },
            None => return EndOfFile
          }
        }
      }

      return Char(chr);
    }
    return EndOfFile;
  };
}

fn main() {
  let (tokenPort, tokenChan) = Chan::new();

  spawn(readTokens(tokenChan));
  let mut parser = Parser {tokenInput: tokenChan, currentToken: Char(' ')};
}
