#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct InputSymbol {
    pub name: String, // Textual representation of the symbol
}
pub const EPSILON_SYMBOLS: [&str; 5] = ["epsilon", "ε", "ϵ", "Є", "$"];
pub const EPSILON: &str = EPSILON_SYMBOLS[0];

impl InputSymbol {
    /// Create a new InputSymbol
    pub fn new(name: &str) -> Self {
        InputSymbol {
            name: name.to_string(),
        }
    }

    /// Create a new InputSymbol from a String
    pub fn from_string(name: String) -> Self {
        InputSymbol { name }
    }

    /// Get the name of the variable
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Display the variable as its name
    pub fn display(&self) -> String {
        self.name.clone()
    }
}

pub fn epsilon() -> InputSymbol {
    InputSymbol::new(EPSILON)
}

pub fn char_to_symbol(i: u8) -> InputSymbol {
    InputSymbol {
        name: if (0x20..=0x7e).contains(&i) {
            (i as char).to_string()
        } else {
            format!("\\x{:02x}", i)
        },
    }
}

pub fn symbol_to_char(symbol: &InputSymbol) -> Option<u8> {
    if symbol.name.len() == 1 {
        Some(symbol.name.as_bytes()[0])
    } else if symbol.name.starts_with("\\x") {
        u8::from_str_radix(&symbol.name[2..], 16).ok()
    } else {
        None
    }
}

fn _main() {
    // Example usage
    let start_variable = InputSymbol::new("S");
    let another_variable = InputSymbol::new("A");

    // Print variables
    println!("Start InputSymbol: {}", start_variable.display());
    println!("Another InputSymbol: {}", another_variable.display());

    // Test equality
    let yet_another = InputSymbol::new("A");
    println!(
        "Are the two terminals equal? {}",
        another_variable == yet_another
    );
}
