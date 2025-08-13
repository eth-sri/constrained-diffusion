import difflib
import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from datasets import load_dataset
import regex

from constrained_diffusion.cfgs.cpp import (
    cpp_grammar_preprocessed,
    CPP_float_number,
    cpp_grammar,
)
from constrained_diffusion.constrain_utils import (
    lex,
    prelex_word,
    EOS,
    generated_language,
    compile_lex_map,
    derive_supertokens,
    language_from_words,
    language_from_program_with_gaps,
    reconstruct_word_boundaries,
)


@pytest.mark.parametrize(
    "s",
    [
        "0.0",
        "1.0",
        # "-1.0",
        "1.0e10",
        "1.0E-10",
        "1.0e+10",
        "1.0E+10",
        "1.0e-10",
        "1.0E-10",
        "12345678901234567890.12345678901234567890e+12345678901234567890",
        "12345678901234567890.12345678901234567890E+12345678901234567890",
        ".20",
        "0.20",
        "0.20e10",
        "1e-10",
        "1.0e-10",
        "1e10",
        # "-1",
        # "-1000",
    ],
)
def test_cpp_float_correct(s: str):
    float_regex = CPP_float_number
    s = prelex_word(s, "\x02\x03", is_first=True, is_last=True)
    regex_pattern = regex.compile(float_regex)
    assert regex_pattern.match(
        s
    ), f"String '{s}' does not match the float regex '{float_regex}'"


simple_examples = [
    """#include <vector>
#include <numeric>

int sum_even(const std::vector<int>& nums) {
    return std::accumulate(nums.begin(), nums.end(), 0, [](int sum, int num) { return sum + num % 2; });
}""",
    """
    #include <iostream>
using namespace std;

int main() {

    switch (day) {
        case 1:
            cout << "Monday" << endl;
            break;
        default:
            cout << "Invalid input! Please enter a number between 1 and 7." << endl;
    }

    return 0;
}""",
    """
bool is_prime(long n){
    return true;
}""",
    """
string sort_numbers(string numbers){
    map<string,int> tonum={{"zero",0},{"one",1},{"two",2},{"three",3},{"four",4},{"five",5},{"six",6},{"seven",7},{"eight",8},{"nine",9}};
    map<int,string> numto={{0,"zero"},{1,"one"},{2,"two"},{3,"three"},{4,"four"},{5,"five"},{6,"six"},{7,"seven"},{8,"eight"},{9,"nine"}};
    int count[10];
    return out;
}
    """,
    """
int count_distinct_characters(string str){ 
    return ::tolower(str);
}
    """,
    """
string string_xor(string a,string b){
    return a and b;
}
    """,
    """
     string string_xor(string a,string b){
                      return '0';
         }
         """,
    """int main() {
    return 0;}
    """,
    """int main() {
    // Simple C++ example that returns 0
    return 0;}
    """,
    """int main() {
    return i.hello();
    }
    """,
    """
    /*Check if in given vector of numbers, are any two numbers closer 
    to each other than*/
    int main() {
    // Simple C++ example that returns 0
    return 0;}
    """,
    """
    bool is_palindrome(string str){
    string s(str.rbegin(),str.rend());
    return s==str;
}
""",
    """
    bool foo(){
    string word;
    stringstream ss;
    while (ss >> word) {
        words.push_back(word);
    }
    }
    """,
    """
/*
Check if in given vector of numbers, are any two numbers closer to each other than
given threshold.
>>> has_close_elements({1.0, 2.0, 3.0}, 0.5)
false
>>> has_close_elements({1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3)
true
*/
bool has_close_elements(vector<float> numbers, float threshold){
    int i,j;

    return false;
}
""",
    """
bool has_close_elements(float numbers, float threshold){
    int i;
    if (i < j){
        return true;
    }

    return false;
}
""",
    """
bool has_close_elements(vector<float> numbers, float threshold){
    int i,j;
    if (abs(numbers[i]-numbers[j])<threshold){
        return true;
    }

    return false;
}
""",
    """
#include<stdio.h>
#include<vector>
#include<math.h>
bool has_close_elements(vector<float> numbers, float threshold){
    int i = 0;
    i = 1;

    return false;
}
""",
    """
bool has_close_elements(vector<float> numbers, float threshold){
    for (int i=0;i<numbers;i++) {
        return true;
    }

    return false;
}
""",
    """
bool has_close_elements(vector<float> numbers, float threshold){
    int i,j;

    for (i=0;i<numbers.size();i++)
    if (abs(numbers[i]-numbers[j])<threshold)
    return true;

    return false;
}
""",
    """
/*
Check if in given vector of numbers, are any two numbers closer to each other than
given threshold.
>>> has_close_elements({1.0, 2.0, 3.0}, 0.5)
false
>>> has_close_elements({1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3)
true
*/
#include<stdio.h>
#include<vector>
#include<math.h>
using namespace std;
bool has_close_elements(vector<float> numbers, float threshold){
    int i,j;
    
    for (i=0;i<numbers.size();i++)
    for (j=i+1;j<numbers.size();j++)
    if (abs(numbers[i]-numbers[j])<threshold)
    return true;

    return false;
}
""",
    """
#include<stdio.h>
#include<vector>
#include<math.h>
using namespace std;
bool has_close_elements(vector<float> numbers, float threshold){
    int i,j;
    
    for (i=0;i<numbers.size();i++)
    for (j=i+1;j<numbers.size();j++)
    if (abs(numbers[i]-numbers[j])<threshold)
    return true;

    return false;
}""",
]


@pytest.mark.parametrize(
    "example",
    enumerate(simple_examples),
)
def test_cpp_simple_examples(example):
    i, example = example
    program = example
    grammar, lexing = cpp_grammar_preprocessed()
    program = prelex_word(program, "\x02\x03", is_first=True, is_last=True)
    lexed = lex(program, lexing, is_first=True)
    assert any(
        grammar.accepts(lexied[0])
        for lexied in lexed
        if not lexied[1] and not lexied[2]
    ), "Failed for {} with grammar\n{}\n{}\n{}\n{}".format(
        program,
        grammar.to_text(),
        len(lexed),
        '\n'.join(str(lexied) for lexied in lexed if not lexied[1] and not lexied[2]),
        lexing
    )


def cpp_compiles(ts_program, timeout=300) -> str:
    with NamedTemporaryFile(suffix=".cpp") as f:
        f.write(ts_program.encode())
        f.flush()
        try:
            res = subprocess.run(
                [
                    "g++",
                    "-fsyntax-only",
                    f.name,
                ],
                capture_output=True,
                timeout=timeout,
            )
            if res.returncode != 0:
                return res.stderr.decode()
            return res.stderr.decode()
        except subprocess.TimeoutExpired:
            return "Timeout"
        except subprocess.CalledProcessError as e:
            return e.stderr.decode()


def syntax_error(ts_program):
    """
    Check if the given C++ program has a syntax error.
    """
    syntax_error_patterns = [
        r"expected",
        r"unexpected",
        r"missing terminating",
        r"stray",
        r"unterminated",
        r"unmatched",
        r"multi-character character constant",
    ]
    compilable = cpp_compiles(ts_program)
    if any(pattern in compilable for pattern in syntax_error_patterns):
        return True
    return False


@pytest.mark.parametrize(
    "instance",
    list(
        (x["task_id"], x)
        for x in load_dataset("THUDM/humaneval-x", "cpp", split="test", trust_remote_code=True)
    ),
)
def test_cpp_dataset(instance):
    i, instance = instance
    orig_program: str = instance["declaration"] + instance["canonical_solution"]
    # strip away the preceding /* comment */
    error = syntax_error(orig_program)
    # remove everything from and after the line starting with "#undef"
    orig_program = orig_program.split("\n#undef")[0]

    grammar, lexing = cpp_grammar_preprocessed()
    prelexed_program = prelex_word(
        orig_program, "\x02\x03", is_first=True, is_last=True
    )
    lexed = lex(prelexed_program, lexing, is_first=True)
    assert (
        any(
            grammar.accepts(lexied[0])
            for lexied in lexed
            if not lexied[1] and not lexied[2]
        )
        != error
    ), (
        "Failed for {}\n{}\n{}\n{}\n{}".format(
      instance['task_id'],
            orig_program,
        len(lexed),
        '\n'.join(str(x) for x in lexed),
        lexing,)
    )


def test_instance0():
    orig_program = [
        """
    #include<stdio.h>
#include<vector>
#include<math.h>
using namespace std;
#include<algorithm>
#include<stdlib.h>
bool has_close_elements(vector<float> numbers, float threshold){
    list =  numbers;
    fire = min_index = 0;
      2 = 1;
    first = 1;
    /* ( i 0;
{
        if
true){
        for
            if (
1 0;
    }""",
        ((("comment",), True, True)),
        """  false
}""",
        EOS,
    ]

    grammar, lex_map, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    prelexed_program = [
        prelex_word(w, "\x02\x03", is_first=i == 0, is_last=i == len(w))
        if isinstance(w, str)
        else w
        for i, w in enumerate(orig_program)
    ]
    language = generated_language(
        prelexed_program,
        lex_map,
        grammar.get_terminals(),
        subtokens=subtokens,
        supertokens=derive_supertokens(subtokens),
    )
    assert language.is_empty()


def test_instance1():
    orig_program = [
        """
    #include<stdio.h>
#include<vector>
#include<math.h>
using namespace std;
#include<algorithm>
#include<stdlib.h>
bool has_close_elements(vector<float> numbers, float threshold){
    list =  numbers;
    fire = min_index = 0;
      2 = 1;
    first = 1;
    /* ( i 0;
{
        if
true){
        for
            if (
1 0;
    }""",
        ((("comment",), True, True)),
        """  false
}""",
        EOS,
    ]

    grammar, lex_map, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    prelexed_program = [
        prelex_word(w, "\x02\x03", is_first=i == 0, is_last=i == len(w))
        if isinstance(w, str)
        else w
        for i, w in enumerate(orig_program)
    ]
    language = generated_language(
        prelexed_program,
        lex_map,
        grammar.get_terminals(),
        subtokens=subtokens,
        supertokens=derive_supertokens(subtokens),
    )
    assert language.is_empty()


def test_instance148():
    orig_program = [
        """
        #include<stdio.h>
#include<math.h>
#include<vector>
#include<string>
#include<algorithm>
using namespace std;
#include<stdlib.h>
vector<string> bf(string planet1,string planet2){

vector<string> result;
if(planet1=="Jupiter" && planet2=="Neptune"){
result.push_back("Saturn");
result.push_back("Uranus");
}
 if(planet1=="Earth" && planet2=="Mercury"){
result.push_back("Venus");
}
 if(planet1=="Mercury" && planet2=="Uranus"){
result.push_back("Venus");
result.push_back("Earth");
result.push_back("Mars");
result.push_back("Jupiter");
result.push_back("Saturn");
}
}"""
    ]

    grammar, lex_map, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    prelexed_program = [
        prelex_word(w, "\x02\x03", is_first=i == 0, is_last=i == len(w))
        if isinstance(w, str)
        else w
        for i, w in enumerate(orig_program)
    ]
    language = generated_language(
        prelexed_program,
        lex_map,
        grammar.get_terminals(),
        subtokens=subtokens,
        supertokens=derive_supertokens(subtokens),
    )
    assert not grammar.is_intersection_empty(language, 100), (
        f"Failed for {orig_program}\n" f"{grammar.to_text()}\n" f"{language}"
    )


def test_empty_expression():
    orig_program = [
        """
        vector<int> intersperse(vector<int> numbers, int delimeter){ 

    vector<int> result;
    for (;umbers.empty();) {
        result.push_back(numbers.back());
        result.push_back(delimeter);
        numbers.pop_back();
    }
    return result;
}
"""
    ]

    grammar, lex_map, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    prelexed_program = [
        prelex_word(w, "\x02\x03", is_first=i == 0, is_last=i == len(w))
        if isinstance(w, str)
        else w
        for i, w in enumerate(orig_program)
    ]
    language = generated_language(
        prelexed_program,
        lex_map,
        grammar.get_terminals(),
        subtokens=subtokens,
        supertokens=derive_supertokens(subtokens),
    )
    assert not grammar.is_intersection_empty(language, 100), (
        f"Failed for {orig_program}\n" f"{grammar.to_text()}\n" f"{language}"
    )


@pytest.mark.skip()
def test_6():
    orig_program = [
        """
vector<int> parse_nested_parens(string paren_string){
    vector<int> result;
    int i;
    int level = 0;
    stringstream ss(paren_string);
    string word;
    while(ss  >> word){
        if(word == "("){
            level++;
        }else if(word == ")"){
            level--;
        }
    }
    for(int i = 0; i < result.size(); i++){
        if(result[i] < level){
            // this is not a syntactic error! could be an initializer for an array of name result of type level, size i
            level result[i];
        }
    }
    result.push_back(level);
    return result;
}
"""
    ]

    grammar, lex_map, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    prelexed_program = [
        prelex_word(w, "\x02\x03", is_first=i == 0, is_last=i == len(w))
        if isinstance(w, str)
        else w
        for i, w in enumerate(orig_program)
    ]
    language = generated_language(
        prelexed_program,
        lex_map,
        grammar.get_terminals(),
        subtokens=subtokens,
        supertokens=derive_supertokens(subtokens),
    )
    assert grammar.is_intersection_empty(language, 100), (
        f"Failed for {orig_program}\n" f"{grammar.to_text()}\n" f"{language}"
    )


@pytest.mark.skip()
def test_lambda_gap():
    orig_program = [
        """#include <vector>
    #include <numeric>
    
    int sum_even(const std::vector<int>& nums) {
        return std::accumulate(nums.begin(), """,
        None,
        """.end(), 0, [](int sum, int num) { return sum + num % 2 });
    }""",
        EOS,
    ]

    grammar, lex_map, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    prelexed_program = [
        prelex_word(w, "\x02\x03", is_first=i == 0, is_last=i == len(w))
        if isinstance(w, str)
        else w
        for i, w in enumerate(orig_program)
    ]
    language = generated_language(
        prelexed_program,
        lex_map,
        grammar.get_terminals(),
        subtokens=subtokens,
        supertokens=derive_supertokens(subtokens),
        trace=True,
    )
    # ONLY if there are no comments allowed unfortunately
    assert grammar.is_intersection_empty(language, 100), (
        f"Failed for {orig_program}\n" f"{grammar.to_text()}\n" f"{language}"
    )


def test_99():
    orig_program = [
        """int closest_integer(string value){
    double number = atof(value.c_str());
    int integer = (int) number;
    if(integer % 2 != 0 && number < 0){
        integer += 1;
    }
    if(fabs(integer - number) == 0.5){
        integer += 1;
    }
    return integer;
}""",
        EOS,
    ]

    grammar, lex_map, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    prelexed_program = [
        prelex_word(w, "\x02\x03", is_first=i == 0, is_last=i == len(w))
        if isinstance(w, str)
        else w
        for i, w in enumerate(orig_program)
    ]
    language = generated_language(
        prelexed_program,
        lex_map,
        grammar.get_terminals(),
        subtokens=subtokens,
        supertokens=derive_supertokens(subtokens),
        trace=True,
    )
    assert not grammar.is_intersection_empty(language, 100), (
        f"Failed for {orig_program}\n" f"{grammar.to_text()}\n" f"{language}"
    )


def test_99_2():
    orig_program = [
        """int closest_integer(string value){
    double a;
    a = atof(value.c_str());
    int floorA = floor(a);
    int ceilingA = ceil(a);
    int floorDiff = abs(floorA - a);
    int ceilingDiff = abs(ceilingA - a);
    if(floorDiff == ceilingDiff){
        return (floorA < 0 && floorA != 0)?ceil(a):floor(a);
    }else if(floorDiff < ceilingDiff){
        return floor(a);
    }else{
        return ceil(a);

}
""",
        EOS,
    ]

    grammar, lex_map, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    prelexed_program = [
        prelex_word(w, "\x02\x03", is_first=i == 0, is_last=i == len(w))
        if isinstance(w, str)
        else w
        for i, w in enumerate(orig_program)
    ]
    language = generated_language(
        prelexed_program,
        lex_map,
        grammar.get_terminals(),
        subtokens=subtokens,
        supertokens=derive_supertokens(subtokens),
        trace=True,
    )
    assert grammar.is_intersection_empty(language, 100), (
        f"Failed for {orig_program}\n" f"{grammar.to_text()}\n" f"{language}"
    )


def test_find_valid_completion():
    orig_program = [
        """#include <vector>
    #include <numeric>

    int sum_even(const std::vector<int>& nums) {
        return std::accumulate(nums.begin(), """,
        None,
        """.end(), 0, [](int sum, int num) { return sum + num % 2 });
    }""",
        EOS,
    ]

    grammar, lex_map_raw, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map_raw, subtokens=subtokens)
    grammar = grammar.to_normal_form()
    prelexed_program = [
        prelex_word(w, "\x02\x03", is_first=i == 0, is_last=i == len(w))
        if isinstance(w, str)
        else w
        for i, w in enumerate(orig_program)
    ]
    language = generated_language(
        prelexed_program,
        lex_map,
        grammar.get_terminals(),
        subtokens=subtokens,
        supertokens=derive_supertokens(subtokens),
        trace=True,
    )
    word = grammar.example_word(language, 100)
    byte_level_lang = language_from_words(word.split(" "), lex_map_raw, subtokens)
    print("Example bytes", byte_level_lang.example_word())
    full_lang = language_from_program_with_gaps(prelexed_program, False, True)
    print("Example full", full_lang.example_word())
    complete_example = byte_level_lang.intersection(full_lang).example_word()
    print(f"Example word: {complete_example}")


@pytest.mark.parametrize(
    "string",
    [
        "//////////////////////////",
        "******************************************",
        "+++++++++++++++++++++++++++++++++++++++++++",
        "-------------------------------------------",
    ],
)
@pytest.mark.timeout(30)
def test_no_blowup(string):
    grammar, lex_map_raw, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map_raw, subtokens=subtokens)
    prelexed_program = prelex_word(string, "\x02\x03", is_first=False, is_last=False)
    res = lex(prelexed_program, lex_map, is_first=False)
    print(res)
    assert len(res) <= 200


@pytest.mark.parametrize(
    "code",
    [
        """#\u0002include\u0003<\u0002stdio\u0003.\u0002h\u0003>\n#\u0002include\u0003<\u0002math\u0003.\u0002h\u0003>\n#\u0002include\u0003<\u0002vector\u0003>\n\u0002using\u0003 \u0002namespace\u0003 \u0002std\u0003;\n#\u0002include\u0003<\u0002algorithm\u0003>\n#\u0002include\u0003<\u0002stdlib\u0003.\u0002h\u0003>\n\u0002int\u0003 \u0002count_nums\u0003(\u0002vector\u0003<\u0002int\u0003> \u0002n\u0003){\n    \u0002int\u0003 \u0002count\u0003 = \u00020\u0003;\n    \u0002for\u0003(\u0002int\u0003 \u0002i\u0003 :\u00020\u0003){\n        \u0002if\u0003(\u0002sum\u0003)\u0002return\u0003;\u0002else\u0003\u0002switch\u0003([]{}){\u0002case\u0003[]{}://)(\u0002i\u0003 \u00020\u0003){\n            \u0002count\u0003++;\n    }\n       }\n \u0002count\u0003;}\n""",
    ],
)
def test_reconstruct_valid_completion_boundaries(code):
    grammar, lex_map_raw, subtokens = cpp_grammar()
    lex_map = compile_lex_map(lex_map_raw, subtokens=subtokens)
    grammar = grammar.to_normal_form()

    lexed_code = lex(code, lex_map, is_first=True)
    assert any(
        grammar.accepts(lexied[0])
        for lexied in lexed_code
        if not lexied[1] and not lexied[2]
    ), f"Failed to reconstruct valid completion boundaries for {code} with grammar"

    reconstructed_program = reconstruct_word_boundaries(code)
    prelexed_reconstructed_program = prelex_word(
        reconstructed_program, "\x02\x03", is_first=True, is_last=True
    )
    print(
        "\n".join(
            repr(x)
            for x in difflib.unified_diff(
                code.splitlines(), prelexed_reconstructed_program.splitlines()
            )
        )
    )
    lexed_prelexed_reconstructed_program = lex(
        prelexed_reconstructed_program, lex_map, is_first=True
    )
    assert any(
        grammar.accepts(lexied[0])
        for lexied in lexed_prelexed_reconstructed_program
        if not lexied[1] and not lexied[2]
    ), f"Failed to reconstruct valid completion boundaries for {reconstructed_program} with grammar"


if __name__ == "__main__":
    pytest.main([__file__])
