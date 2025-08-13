regex_dfa
=========

> This is a fork of [regex_dfa](https://github.com/jneem/regex-dfa), adapted to compile with the latest Rust version and to work with a newer `regex` crate.

A crate for compiling regular expressions down to deterministic finite
automata.

# Why?

Some regular expression implementations (e.g. [rust's regex
library](http://github.com/rust-lang/regex)) are based on simulating
non-deterministic finite automata (NFAs). By turning NFAs into DFAs, we can
get a speed boost, at the cost of some compilation time and memory usage.
[Preliminary benchmarks](http://bl.ocks.org/jneem/raw/3f08ade195796358d027/?data=%5B%7B%22x%22%3A%201506622%2C%20%22y%22%3A%201646883%2C%20%22bench%22%3A%20%22%28%3Fi%29Twain%22%2C%20%22ratio%22%3A%200.914832444077691%7D%2C%20%7B%22x%22%3A%2015187577%2C%20%22y%22%3A%20125978857%2C%20%22bench%22%3A%20%22%5Ba-q%5D%5B%5Eu-z%5D%7B13%7Dx%22%2C%20%22ratio%22%3A%200.12055655497811034%7D%2C%20%7B%22x%22%3A%201630324%2C%20%22y%22%3A%201631615%2C%20%22bench%22%3A%20%22Tom%7CSawyer%7CHuckleberry%7CFinn%22%2C%20%22ratio%22%3A%200.9992087594193483%7D%2C%20%7B%22x%22%3A%201501634%2C%20%22y%22%3A%20243419316%2C%20%22bench%22%3A%20%22.%7B0%2C2%7D%28Tom%7CSawyer%7CHuckleberry%7CFinn%29%22%2C%20%22ratio%22%3A%200.0061689188215449595%7D%2C%20%7B%22x%22%3A%201506470%2C%20%22y%22%3A%20295351074%2C%20%22bench%22%3A%20%22.%7B2%2C4%7D%28Tom%7CSawyer%7CHuckleberry%7CFinn%29%22%2C%20%22ratio%22%3A%200.005100607827821882%7D%2C%20%7B%22x%22%3A%201583452%2C%20%22y%22%3A%201724976%2C%20%22bench%22%3A%20%22Tom.%7B10%2C25%7Driver%7Criver.%7B10%2C25%7DTom%22%2C%20%22ratio%22%3A%200.9179559599669792%7D%2C%20%7B%22x%22%3A%201290763%2C%20%22y%22%3A%207366270%2C%20%22bench%22%3A%20%22%5B%5C%22%27%5D%5B%5E%5C%22%27%5D%7B0%2C30%7D%5B%3F%21%5C%5C.%5D%5B%5C%22%27%5D%22%2C%20%22ratio%22%3A%200.17522613208584534%7D%2C%20%7B%22x%22%3A%201571131%2C%20%22y%22%3A%20137662852%2C%20%22bench%22%3A%20%22%28%3Fi%29Tom%7CSawyer%7CHuckleberry%7CFinn%22%2C%20%22ratio%22%3A%200.011412890094707612%7D%2C%20%7B%22x%22%3A%201493996%2C%20%22y%22%3A%20131818244%2C%20%22bench%22%3A%20%22%28%5BA-Za-z%5Dawyer%7C%5BA-Za-z%5Dinn%29%5C%5Cs%22%2C%20%22ratio%22%3A%200.011333757412213746%7D%2C%20%7B%22x%22%3A%201499437%2C%20%22y%22%3A%2094501284%2C%20%22bench%22%3A%20%22%5C%5Cb%5C%5Cw%2Bnn%5C%5Cb%22%2C%20%22ratio%22%3A%200.015866842613482375%7D%2C%20%7B%22x%22%3A%201494150%2C%20%22y%22%3A%201620677%2C%20%22bench%22%3A%20%22Huck%5Ba-zA-Z%5D%2B%7CSaw%5Ba-zA-Z%5D%2B%22%2C%20%22ratio%22%3A%200.9219295393221475%7D%2C%20%7B%22x%22%3A%204604887%2C%20%22y%22%3A%2084880712%2C%20%22bench%22%3A%20%22%5C%5Cs%5Ba-zA-Z%5D%7B0%2C12%7Ding%5C%5Cs%22%2C%20%22ratio%22%3A%200.0542512767800534%7D%2C%20%7B%22x%22%3A%201657821%2C%20%22y%22%3A%201664874%2C%20%22bench%22%3A%20%22%5Ba-z%5Dshing%22%2C%20%22ratio%22%3A%200.9957636433748139%7D%2C%20%7B%22x%22%3A%20100143%2C%20%22y%22%3A%2099739%2C%20%22bench%22%3A%20%22Twain%22%2C%20%22ratio%22%3A%201.0040505719929014%7D%2C%20%7B%22x%22%3A%204877680%2C%20%22y%22%3A%2071127171%2C%20%22bench%22%3A%20%22%5Ba-zA-Z%5D%2Bing%22%2C%20%22ratio%22%3A%200.06857688744572732%7D%2C%20%7B%22x%22%3A%201584255%2C%20%22y%22%3A%2062666721%2C%20%22bench%22%3A%20%22%5C%5CbF%5C%5Cw%2Bn%5C%5Cb%22%2C%20%22ratio%22%3A%200.025280642974761677%7D%5D)
show a substantial speed improvement over rust's default `regex` crate.

# Limitations

- Turning an NFA into a DFA can take a lot of memory, especially when unicode character classes are involved.
- Subgroup captures are a bit tricky, and this crate does not handle them.

# License

`regex_dfa` is distributed under the MIT license and the Apache license (version 2.0).
See LICENSE-APACHE and LICENSE-MIT for details.

