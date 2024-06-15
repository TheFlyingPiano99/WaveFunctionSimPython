from typing import Dict
from colorama import Fore, Back, Style


def try_read_param(config: Dict, tokens: str, default_val, reached_tokens: str = ""):
    tokens = tokens.split(".")
    for token in tokens:
        try:
            config = config[token]
        except KeyError:
            print(Fore.RED
                  + f"Did not found token \"{token}\" under \"{reached_tokens}\"!\n"
                  + f"Falling back to the default value of {default_val}.\n"
                  + Style.RESET_ALL)
            return default_val
        reached_tokens += "." + token
    return config  # At this point config should contain the value
