import re

# 1SS marks isolated capitals: "The NASA Cat" -> "\x02the NASA \x02cat"
class OneShotShiftTransformer:
    # hack: 1SS reuses the repeat-key slot (\x02, the semicolon position). passing
    # rpt_key=True to build_layout_and_engine just loads the layout where that key
    # exists; there is no actual repeat-key behavior anywhere in 1SS.
    ONE_SHOT_CHAR = "\x02"

    def __init__(self, one_shot_char: str = ONE_SHOT_CHAR):
        self.one_shot_char = one_shot_char
        # an uppercase letter not adjacent to another uppercase (i.e. isolated)
        self.pattern = re.compile(r"(?<![A-Z])([A-Z])(?![A-Z])")

    def transform(self, text: str) -> str:
        if not text:
            return text
            
        def replace_match(match):
            char = match.group(1)
            return self.one_shot_char + char.lower()
            
        return self.pattern.sub(replace_match, text)
