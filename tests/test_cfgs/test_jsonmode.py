import json

import jsonschema
import pytest
from datasets import load_dataset
from jsonschema.validators import validator_for
from transformers import AutoTokenizer

from constrained_diffusion.constrain_utils import (
    compile_lex_map,
    lex,
    generated_language,
    preprocessed_generate_stuff,
)

from rustformlang.fa.bytes_dfa import regex_to_dfa
from constrained_diffusion.cfgs.jsonschema import schema_to_cfg


def test_schema_to_cfg_2():
    schmema = {
        "$id": "https://example.com/fstab",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["/"],
        "properties": {
            "/": {
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "mount_point": {"type": "string"},
                    "file_system_type": {"type": "string"},
                    "options": {"type": "string"},
                    "dump": {"type": "string", "pattern": "^[0-1]$"},
                    "pass": {"type": "string", "pattern": "^[0-2]$"},
                },
                "required": [
                    "device",
                    "mount_point",
                    "file_system_type",
                    "options",
                    "dump",
                    "pass",
                ],
            }
        },
        "patternProperties": {
            "^(/[^/]+)+$": {
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "mount_point": {"type": "string"},
                    "file_system_type": {"type": "string"},
                    "options": {"type": "string"},
                    "dump": {"type": "string", "pattern": "^[0-1]$"},
                    "pass": {"type": "string", "pattern": "^[0-2]$"},
                },
                "required": [
                    "device",
                    "mount_point",
                    "file_system_type",
                    "options",
                    "dump",
                    "pass",
                ],
            }
        },
        "additionalProperties": False,
    }
    _ = schema_to_cfg(schmema)


def test_schema_string():
    schema = """{"title": "PromotionalCampaign", "type": "object", "properties": {"campaignID": {"title": "Campaign ID", "type": "string"}, "productID": {"title": "Product ID", "type": "string"}, "startDate": {"title": "Start Date", "type": "string", "format": "date"}, "endDate": {"title": "End Date", "type": "string", "format": "date"}, "discountDetails": {"title": "Discount Details", "type": "string"}}, "required": ["campaignID", "productID", "startDate", "endDate"]}"""
    _ = schema_to_cfg(json.loads(schema))


def test_schema_int():
    schema = """{"title": "UserReview", "type": "object", "properties": {"reviewId": {"title": "Review ID", "type": "string"}, "productId": {"title": "Product ID", "type": "string"}, "reviewer": {"title": "Reviewer", "type": "object", "properties": {"userId": {"title": "User ID", "type": "string"}, "name": {"title": "Name", "type": "string"}}, "required": ["userId", "name"]}, "rating": {"title": "Rating", "type": "integer", "minimum": 1, "maximum": 5}, "comments": {"title": "Comments", "type": "string"}}, "required": ["reviewId", "productId", "reviewer", "rating", "comments"]}"""
    _ = schema_to_cfg(json.loads(schema))


def test_schema_complex():
    schema = """{"title": "WirelessAccessPoint", "type": "object", "properties": {"ssid": {"title": "SSID", "type": "string"}, "securityProtocol": {"title": "SecurityProtocol", "type": "string"}, "bandwidth": {"title": "Bandwidth", "type": "string"}}, "required": ["ssid", "securityProtocol", "bandwidth"]}"""
    grammar, lexing, subtokens = schema_to_cfg(json.loads(schema))
    lex_map = compile_lex_map(lexing, subtokens)
    lexed = lex(
        '{"ssid": "OfficeNetSecure", "securityProtocol": "WPA2-Enterprise", "bandwidth": "1300 Mbps"}',
        lex_map,
    )
    print(grammar.to_text())
    print(next(iter(lexed))[0])
    assert any(lexied[0] in grammar for lexied in lexed)
    lexed = lex(
        '{"ssid": "OfficeNetSecure", "bandwidth": "1300 Mbps"}',
        lex_map,
    )
    assert not any(lexied[0] in grammar for lexied in lexed)


def test_schema_complex_removing_ambiguity():
    schema = """{"title": "WirelessAccessPoint", "type": "object", "properties": {"ssid": {"title": "SSID", "type": "string"}, "securityProtocol": {"title": "SecurityProtocol", "type": "string"}, "bandwidth": {"title": "Bandwidth", "type": "string"}}, "required": ["ssid", "securityProtocol", "bandwidth"]}"""
    grammar, lexing, subtokens = schema_to_cfg(json.loads(schema))
    lex_map = compile_lex_map(lexing, subtokens)
    lexed = lex(
        '{"ss',
        lex_map,
    )
    assert len(lexed) == 1
    lexed = lex(
        '{"ssid": "office',
        lex_map,
    )
    assert len(lexed) == 1


def test_schema_object_optional():
    schema = """{"title": "WirelessAccessPoint", "type": "object", "properties": {"ssid": {"title": "SSID", "type": "string"}, "securityProtocol": {"title": "SecurityProtocol", "type": "string"}, "bandwidth": {"title": "Bandwidth", "type": "string"}}, "required": ["ssid"]}"""
    grammar, lexing, subtokens = schema_to_cfg(json.loads(schema))
    word = '{"ssid": "OfficeNetSecure", "securityProtocol": "WPA2-Enterprise", "bandwidth": "1300 Mbps"}'
    lexed = lex(
        word,
        compile_lex_map(lexing, subtokens),
    )
    assert any(
        lexied[0] in grammar for lexied in lexed
    ), f"Failed for {word}.\n Grammar {grammar.to_text()}\n. Lexed: {lexed}"


def test_schema_array():
    schema = (
        """{"title": "ProductList", "type": "array", "items": {"type": "string"}}"""
    )
    grammar, lexing, subtokens = schema_to_cfg(json.loads(schema))
    compiled_lex_map = compile_lex_map(lexing, subtokens)
    lexed = lex(
        '["Laptop", "Smartphone", "Tablet"]',
        compiled_lex_map,
    )
    assert any(grammar.accepts(lexied[0]) for lexied in lexed)
    lexed = lex(
        '[1, "Smartphone", "Tablet"]',
        compiled_lex_map,
    )
    assert not any(grammar.accepts(lexied[0]) for lexied in lexed)


def test_schema_one_of():
    schema = """{"oneOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}, {"items": {"type": "integer"}, "type": "array"}]}"""
    grammar, lexing, subtokens = schema_to_cfg(json.loads(schema))
    for word in [
        "123",
        '"hello"',
        "null",
        "[1, 2, 3]",
    ]:
        lexed = lex(
            word,
            compile_lex_map(lexing, subtokens),
        )
        assert any(
            grammar.accepts(lexied[0]) for lexied in lexed
        ), f"Failed for {word}\n lexed: {lexed}\n {grammar.to_text()}"


def test_complex_schema_one_of():
    schema = """{"type": "object", "properties": {"deviceType": {"type": "string"}}, "required": ["deviceType"], "oneOf": [{"properties": {"deviceType": {"const": "smartphone"}, "brand": {"type": "string"}, "model": {"type": "string"}, "screenSize": {"type": "string"}}}, {"properties": {"deviceType": {"const": "laptop"}, "brand": {"type": "string"}, "model": {"type": "string"}, "processor": {"type": "string"}, "RAMSize": {"type": "string"}}}]}"""
    grammar, lexing, subtokens = schema_to_cfg(json.loads(schema))
    word = '{"deviceType": "laptop", "brand": "Dell", "model": "XPS 13", "processor": "Intel Core i7", "RAMSize": "16GB"}'
    compiled_lex_map = compile_lex_map(lexing, subtokens)
    lexed = lex(word, compiled_lex_map)
    assert any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {word}\nlexed {lexed}\n {grammar.to_text()}"
    word = '{"deviceType": "smartphone", "brand": "Dell", "model": "XPS 13", "processor": "Intel Core i7", "RAMSize": "16GB"}'
    lexed = lex(word, compiled_lex_map)
    # is allowed, other fields are parsed as additional properties
    assert any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {word}\nlexed {lexed}\n {grammar.to_text()}"


def test_no_object_declaration():
    schema = {
        "LogisticsDashboard": {
            "type": "object",
            "properties": {
                "totalShipments": {"title": "Total Shipments", "type": "integer"},
                "onTimeDeliveryRate": {
                    "title": "On Time Delivery Rate",
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
                "averageDeliveryTime": {
                    "title": "Average Delivery Time",
                    "type": "number",
                },
                "pendingShipments": {"title": "Pending Shipments", "type": "integer"},
            },
            "required": [
                "totalShipments",
                "onTimeDeliveryRate",
                "averageDeliveryTime",
                "pendingShipments",
            ],
        }
    }
    try:
        grammar, lexing, subtokens = schema_to_cfg(schema)
        pytest.fail("Schema without object declaration should raise an error")
    except Exception:
        pass


def test_pattern_properties():
    schema = {
        "type": "object",
        "properties": {"/": {"type": "string"}},
        "patternProperties": {
            r"^/[^/]+$": {
                "type": "string",
            }
        },
        "required": ["/"],
        "additionalProperties": False,
    }
    word = {"/": "root", "/mnt": "1TB"}
    jsonschema.validate(word, schema)
    grammar, lexing, subtokens = schema_to_cfg(schema)
    lexed = lex(
        json.dumps(word),
        compile_lex_map(lexing, subtokens),
    )
    assert any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {json.dumps(word)} \n {grammar.to_text()}. lexed: {lexed}. lexing: {lexing}"
    word = {"/": "root", "/mn/t": "1TB"}
    grammar, lexing, subtokens = schema_to_cfg(schema)
    lexed = lex(
        json.dumps(word),
        compile_lex_map(lexing, subtokens),
    )
    assert not any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {json.dumps(word)} \n {grammar.to_text()}. lexed: {lexed}. lexing: {lexing}"


def test_only_pattern_properties():
    schema = {
        "type": "object",
        "patternProperties": {
            r"^/[^/]+$": {
                "type": "string",
            }
        },
        "additionalProperties": False,
    }
    grammar, lexing, subtokens = schema_to_cfg(schema)

    word = {"/mnt": "1TB"}
    jsonschema.validate(word, schema)
    lexed = lex(
        json.dumps(word),
        compile_lex_map(lexing, subtokens),
    )
    assert any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {json.dumps(word)} \n {grammar.to_text()}. lexed: {lexed}. lexing: {lexing}"

    word = {"hello": "1TB"}
    lexed = lex(
        json.dumps(word),
        compile_lex_map(lexing),
    )
    assert not any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {json.dumps(word)} \n {grammar.to_text()}. lexed: {lexed}. lexing: {lexing}"


def test_number_no_duplicate_lex():
    schema = {
        "type": "object",
        "properties": {
            r"value": {
                "type": "number",
            },
            r"value2": {
                "type": "integer",
            },
            r"value3": {
                "type": "integer",
                "minimum": 0,
                "maximum": 9,
            },
        },
        "required": ["value", "value2"],
        "additionalProperties": False,
    }
    grammar, lexing, subtokens = schema_to_cfg(schema)

    word = {"value": 1, "value2": 2, "value3": 3}
    jsonschema.validate(word, schema)
    lexed = lex(
        json.dumps(word),
        compile_lex_map(lexing, subtokens),
    )
    assert len(lexed) == 1
    assert any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {json.dumps(word)} \n {grammar.to_text()}. lexed: {lexed}. lexing: {lexing}"


def test_number_number_integer():
    schema = {
        "title": "LoanApplication",
        "type": "object",
        "properties": {
            "applicantDetails": {
                "type": "object",
                "properties": {
                    "firstName": {"type": "string"},
                    "lastName": {"type": "string"},
                    "dateOfBirth": {"type": "string", "format": "date"},
                    "employmentStatus": {"type": "string"},
                },
                "required": [
                    "firstName",
                    "lastName",
                    "dateOfBirth",
                    "employmentStatus",
                ],
            },
            "loanAmount": {"type": "number"},
            "creditScore": {"type": "integer"},
        },
        "required": ["applicantDetails", "loanAmount", "creditScore"],
    }

    grammar, lexing, subtokens = schema_to_cfg(schema)

    word = {
        "applicantDetails": {
            "firstName": "Sarah",
            "lastName": "Connor",
            "dateOfBirth": "1985-07-12",
            "employmentStatus": "Full-time",
        },
        "loanAmount": 15000,
        "creditScore": 750,
    }
    jsonschema.validate(word, schema)
    lexed = lex(
        json.dumps(word),
        compile_lex_map(lexing, subtokens),
    )
    assert any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {json.dumps(word)} \n {grammar.to_text()}.\n lexed: {lexed}.\n lexing: {lexing}"


def test_get_difference():
    nfa1 = regex_to_dfa(r"ab(c)*")
    nfa2 = regex_to_dfa(r"abc")
    nfa3 = nfa1.difference(nfa2)
    assert not nfa3.is_empty()
    assert nfa3.accepts_string("ab")
    assert not nfa3.accepts_string("abc")
    assert nfa3.accepts_string("abccc")


@pytest.mark.parametrize(
    "instance",
    enumerate(
        list(
            load_dataset(
                "eth-sri/json-mode-eval-cleaned", split="train"
            ).to_iterable_dataset()
        )
    ),
)
def test_jsonmode_dataset(instance):
    i, instance = instance
    schema = json.loads(instance["schema"])
    valid_value = instance["completion"]
    try:
        validator_for(schema).check_schema(schema)
    except jsonschema.exceptions.SchemaError:
        return
    try:
        jsonschema.validate(json.loads(valid_value), schema)
    except jsonschema.exceptions.ValidationError:
        return
    grammar, lexing, subtokens = schema_to_cfg(schema)
    lexed = lex(
        valid_value,
        compile_lex_map(lexing, subtokens=subtokens),
        is_first=True,
    )
    assert any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {valid_value} with grammar\n{grammar.to_text()}\n{lexed}\n{lexing}"


def test_118():
    instance = """
{
  "current_location": "Boulder, Colorado",
  "preferred_units": "Fahrenheit",
  "weather_forecast": "Friday: Sunny, high 75°F, low 50°F. Saturday: Partly cloudy, high 70°F, low 48°F. Sunday: Afternoon showers, high 65°F, low 45°F.",
  "recommended_activities": [
    "Friday: Hiking",
    "Saturday: Short hikes or visiting local parks",
    "Sunday: Indoor activities"
  ]
}"""
    schema = {
        "type": "object",
        "properties": {
            "current_location": {"type": "string"},
            "preferred_units": {"type": "string"},
            "weather_forecast": {"type": "string"},
            "recommended_activities": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "current_location",
            "preferred_units",
            "weather_forecast",
            "recommended_activities",
        ],
    }
    grammar, lexing, subtokens = schema_to_cfg(schema)
    compiled_map = compile_lex_map(lexing, subtokens=subtokens)
    lexed = lex(
        '"Friday: Sunny, high 75°F, low 50°F. Saturday: Partly cloudy, high 70°F, low 48°F. Sunday: Afternoon showers, high 65°F, low 45°F."',
        compiled_map,
        is_first=True,
    )
    print(lexed)
    assert (
        lexed
    ), f"Failed for {instance} with grammar\n{grammar.to_text()}\n{lexed}\n{lexing}"
    lexed = lex(
        instance,
        compiled_map,
        is_first=True,
    )
    assert any(
        grammar.accepts(lexied[0]) for lexied in lexed
    ), f"Failed for {instance} with grammar\n{grammar.to_text()}\n{lexed}\n{lexing}"


@pytest.mark.skip()
def test_gap_lexing():
    tokens = [
        "{\n",
        None,
        '"item',
        None,
        '": "',
        None,
        "-314",
        None,
        '59",',
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    schema = {
        "title": "ConsumableInventoryItem",
        "type": "object",
        "properties": {
            "itemId": {
                "title": "Item ID",
                "type": "string",
                "additionalProperties": False,
            },
            "quantity": {
                "title": "Quantity",
                "type": "integer",
                "minimum": 0,
                "additionalProperties": False,
            },
        },
        "required": ["itemId", "quantity"],
        "additionalProperties": False,
    }
    grammar, lexing, subtokens = schema_to_cfg(schema)
    compiled_map = compile_lex_map(lexing, subtokens=subtokens)
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    single_token_lexing, no_lexing_tokens, supertokens = preprocessed_generate_stuff(
        tokenizer,
        grammar,
        compiled_map,
        trace=False,
        subtokens=subtokens,
        strip_chars=None,
    )
    language = generated_language(
        tokens,
        compiled_map,
        grammar.get_terminals(),
        inject_gap_size=5,
        max_total_injections=5,
        single_token_lexing=single_token_lexing,
        subtokens=subtokens,
        supertokens=supertokens,
        trace=True,
    )
    assert not grammar.is_intersection_empty(
        language, 100
    ), "Generated language is empty, expected it to not be empty"


@pytest.mark.skip()
def test_gap_lexing2():
    """
       {
    .* vehicle .* 123 .* X .{1}
    .* service .*
    """
    tokens = [
        "{\n",
        None,
        None,
        "vehicle",
        None,
        None,
        "123",
        None,
        None,
        "X",
        None,
        "\n",
        None,
        None,
    ]
    schema = {
        "title": "VehicleMaintenanceRecord",
        "type": "object",
        "properties": {
            "vehicleID": {"title": "Vehicle ID", "type": "string"},
            "serviceDate": {
                "title": "Service Date",
                "type": "string",
                "format": "date",
            },
        },
        "required": ["vehicleID", "serviceDate"],
        "additionalProperties": False,
    }
    grammar, lexing, subtokens = schema_to_cfg(schema)
    compiled_map = compile_lex_map(lexing, subtokens=subtokens)
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
    single_token_lexing, no_lexing_tokens, supertokens = preprocessed_generate_stuff(
        tokenizer,
        grammar,
        compiled_map,
        trace=False,
        subtokens=subtokens,
        strip_chars=None,
    )
    single_token_lexing = lex('",', compiled_map, is_first=False)
    language = generated_language(
        tokens,
        compiled_map,
        grammar.get_terminals(),
        inject_gap_size=1,
        max_total_injections=1,
        single_token_lexing=single_token_lexing,
        subtokens=subtokens,
        supertokens=supertokens,
        trace=True,
    )
    assert not grammar.is_intersection_empty(
        language, 100
    ), "Generated language is empty, expected it to not be empty"
