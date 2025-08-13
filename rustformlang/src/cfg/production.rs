use crate::cfg::terminal::Terminal;
use crate::cfg::variable::Variable;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Symbol {
    T(Terminal),
    V(Variable),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Production {
    pub head: Variable,    // The head of the production (e.g., "S", "A")
    pub body: Vec<Symbol>, // The body of the production (e.g., "A b", "a S")
}

impl Production {
    /// Create a new Production
    pub fn new(head: Variable, body: Vec<Symbol>) -> Self {
        Production { head, body }
    }

    /// Display the production as a string
    pub fn display(&self) -> String {
        let body_str: Vec<String> = self
            .body
            .iter()
            .map(|s| match s {
                Symbol::T(t) => t.get_name().to_string(),
                Symbol::V(v) => v.get_name().to_string(),
            })
            .collect();
        format!("{} -> {}", self.head.get_name(), body_str.join(" "))
    }
}

fn _main() {
    // Example usage
    let start_variable = Variable::new("S");
    let another_variable = Variable::new("A");
    let terminal_b = Terminal::new("b");
    let production1 = Production::new(
        start_variable.clone(),
        vec![Symbol::V(another_variable.clone()), Symbol::T(terminal_b)],
    );
    println!("Production 1: {}", production1.display());
}
