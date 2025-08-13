#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct State {
    pub name: String, // The name of the variable (e.g., "S", "A", "B")
}

impl State {
    /// Create a new State
    pub fn new(name: &str) -> Self {
        State {
            name: name.to_string(),
        }
    }

    /// Create a new State from a String
    pub fn from_string(name: String) -> Self {
        State { name }
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

fn _main() {
    // Example usage
    let start_variable = State::new("S");
    let another_variable = State::new("A");

    // Print variables
    println!("Start State: {}", start_variable.display());
    println!("Another State: {}", another_variable.display());

    // Test equality
    let yet_another = State::new("A");
    println!(
        "Are the two states equal? {}",
        another_variable == yet_another
    );
}
