"""Item acknowledgements shared by the OpenAI and xAI realtime dialects."""


def item_matches(sent: dict, actual: dict) -> bool:
    """Keep item fields and ordered parts exact, except unreturned image payloads.

    OpenAI acknowledges input_image by type alone: it never returns the image
    bytes. Apply the same rule to xAI's item dialect without enabling vision.
    Provider-added top-level metadata remains allowed; text, audio and function
    payloads retain the exact comparison. An acknowledgement cannot prove image
    byte identity, but it must preserve the part's position and type.
    """
    if not isinstance(actual, dict):
        return False
    for key, value in sent.items():
        echoed = actual.get(key)
        if key != "content" or not isinstance(value, list):
            if echoed != value:
                return False
            continue
        if not isinstance(echoed, list) or len(echoed) != len(value):
            return False
        for part, echo in zip(value, echoed):
            if isinstance(part, dict) and part.get("type") == "input_image":
                if not isinstance(echo, dict) or echo.get("type") != "input_image":
                    return False
            elif echo != part:
                return False
    return True
