SYSTEM_PROMPT = (
    "You are an expert mechanical engineer. Based on the user's text "
    "requirements, generate the corresponding CAD model design."
)


def prompt_message(prompt: str):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "brep"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
