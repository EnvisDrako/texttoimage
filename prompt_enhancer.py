DEFAULT_QUALITY_APPEND = ", highly detailed, 4k, ultra realistic, sharp focus"
DEFAULT_NEGATIVE = "(low quality), (blurry), text, watermark, extra limb, deformed"

def enhance_prompt(prompt: str, style: str = None, add_quality: bool = True):
    p = prompt.strip()
    if add_quality:
        p = p + DEFAULT_QUALITY_APPEND
    if style:
        p = f"{style}, " + p
    return p

def default_negative_prompt(user_negative: str = ""):
    if user_negative and user_negative.strip():
        return user_negative + ", " + DEFAULT_NEGATIVE
    return DEFAULT_NEGATIVE