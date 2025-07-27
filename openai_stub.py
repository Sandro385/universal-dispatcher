"""A minimal fallback implementation of the OpenAI client.

This module provides a drop‑in replacement for the parts of the official
``openai`` package that are used by the application.  It allows the
application to run in environments where the official package cannot be
installed (e.g. due to restricted network access).  The stub simulates the
behaviour of the OpenAI chat API by echoing the user's message.

Usage:

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        from openai_stub import OpenAI

The real OpenAI package should be installed in production to enable actual
model calls.  When available, Python will import the real package instead
of this stub.
"""

class _Message:
    """Represents a message returned from the OpenAI API."""

    def __init__(self, content: str):
        self.content: str = content
        self.tool_calls: list = []  # the stub never invokes tools


class _Choice:
    """Represents a single choice in the response."""

    def __init__(self, message: _Message):
        self.message = message


class _Completions:
    """Simulates the chat completions endpoint."""

    def create(self, model: str, messages: list, tools=None, tool_choice=None):
        # Use the last message's content as the user input; default to empty
        user_msg = ''
        if messages:
            user_msg = messages[-1].get('content', '')
        reply = f"Echo: {user_msg}"
        return type('Response', (), {
            'choices': [_Choice(_Message(reply))]
        })


class _Chat:
    """Container for the completions API."""

    def __init__(self):
        self.completions = _Completions()


class OpenAI:
    """Fallback OpenAI client class.

    Parameters
    ----------
    base_url: str, optional
        Ignored in this stub; provided for API compatibility.
    api_key: str, optional
        Ignored in this stub; provided for API compatibility.

    This stub stores the provided parameters but does not use them.  The real
    OpenAI client uses the API key and base URL to authenticate and route
    requests; the stub simply echoes messages.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url
        self.api_key = api_key
        self.chat = _Chat()