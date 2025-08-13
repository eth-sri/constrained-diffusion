use crate::input_symbol::{InputSymbol, EPSILON_SYMBOLS};

// Terminals are symbols for CFG
// NOTE: The epsilon symbol *does not exist* in CFGs, it is to be replaced by the empty string (or empty productions)
pub type Terminal = InputSymbol;
pub const TERMINAL_EPSILON_SYMBOLS: [&str; 5] = EPSILON_SYMBOLS;
