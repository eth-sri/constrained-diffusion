use crate::input_symbol::{char_to_symbol, InputSymbol};

pub trait Language {
    fn accepts(&self, input: &Vec<InputSymbol>) -> bool;

    /// This function takes a reference to a finite automaton and a string input,
    /// converts the string into a Vec<InputSymbol>, and checks if the automaton accepts it.
    fn accepts_string(&self, input: &str) -> bool {
        // Convert the string into a Vec of InputSymbol
        let input_symbols: Vec<InputSymbol> = input.bytes().map(char_to_symbol).collect();

        // Use the automaton's `accepts` method to check if it accepts the input
        self.accepts(&input_symbols)
    }
}

pub trait MutLanguage {
    fn accepts(&mut self, input: &Vec<InputSymbol>) -> bool;

    /// This function takes a reference to a finite automaton and a string input,
    /// converts the string into a Vec<InputSymbol>, and checks if the automaton accepts it.
    fn accepts_string(&mut self, input: &str) -> bool {
        // Convert the string into a Vec of InputSymbol
        let input_symbols: Vec<InputSymbol> = input.bytes().map(char_to_symbol).collect();

        // Use the automaton's `accepts` method to check if it accepts the input
        self.accepts(&input_symbols)
    }
}
