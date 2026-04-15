import re


def fix_latex_in_json(json_str: str) -> str:
    """
    Intelligently double-escapes LaTeX backslashes without destroying JSON structural elements.
    """
    # Step A: Protect valid double-backslashes (if the LLM actually escaped correctly)
    s = json_str.replace('\\\\', '__DOUBLE_SLASH__')

    # Step B: Protect valid structural JSON quotes
    s = s.replace('\\"', '__QUOTE__')

    # Step C: The "Tab vs Tau" Solver
    # Protect \n, \t, \r, \b, \f ONLY if they are NOT followed by a letter.
    # (\nu is followed by 'u', so it gets ignored here. \n followed by space gets protected).
    s = re.sub(r'\\([ntrbf])(?![a-zA-Z])', r'__ESC_\1__', s)

    # Step D: Protect valid 4-digit unicode hex escapes (like \u03b2 for beta)
    s = re.sub(r'\\u([0-9a-fA-F]{4})', r'__ESC_U_\1__', s)

    # Step E: All remaining backslashes are guaranteed to be unescaped LaTeX
    # (e.g., \mu, \alpha, \nu, \tau). We double escape them safely!
    s = s.replace('\\', '\\\\')

    # Step F: Restore all protected structural sequences
    s = s.replace('__DOUBLE_SLASH__', '\\\\')
    s = s.replace('__QUOTE__', '\\"')
    s = re.sub(r'__ESC_([ntrbf])__', r'\\\1', s)
    s = re.sub(r'__ESC_U_([0-9a-fA-F]{4})__', r'\\u\1', s)

    return s

