use hashbrown::Equivalent;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Variable {
    pub name: String, // The name of the variable (e.g., "S", "A", "B")
}

impl Variable {
    /// Create a new Variable
    pub fn new(name: &str) -> Self {
        Variable {
            name: name.to_string(),
        }
    }

    /// Create a new Variable from a String
    pub fn from_string(name: String) -> Self {
        Variable { name }
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
    let start_variable = Variable::new("S");
    let another_variable = Variable::new("A");

    // Print variables
    println!("Start Variable: {}", start_variable.display());
    println!("Another Variable: {}", another_variable.display());

    // Test equality
    let yet_another = Variable::new("A");
    println!(
        "Are the two variables equal? {}",
        another_variable == yet_another
    );
}
