from supportcover_rag.config import PromptingConfig
from supportcover_rag.generation import PromptInput, postprocess_generated_text, render_transformer_prompt


class ChatTemplateTokenizer:
    def __init__(self) -> None:
        self.chat_template = "{{ messages }}"
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "<chat-template>"


class PlainTokenizer:
    chat_template = None


def test_render_transformer_prompt_uses_chat_template_when_available() -> None:
    tokenizer = ChatTemplateTokenizer()

    rendered, used_chat_template = render_transformer_prompt(
        tokenizer=tokenizer,
        prompting=PromptingConfig(),
        prompt_input=PromptInput(
            question="Who wrote Hamlet?",
            evidence="Title: Hamlet\n- William Shakespeare wrote Hamlet.",
        ),
    )

    assert rendered == "<chat-template>"
    assert used_chat_template is True
    assert len(tokenizer.calls) == 1
    assert tokenizer.calls[0]["tokenize"] is False
    assert tokenizer.calls[0]["add_generation_prompt"] is True
    rendered_messages = "\n".join(message["content"] for message in tokenizer.calls[0]["messages"])
    assert "<think>" not in rendered_messages
    assert "insufficient evidence" in rendered_messages


def test_render_transformer_prompt_falls_back_to_plain_text() -> None:
    rendered, used_chat_template = render_transformer_prompt(
        tokenizer=PlainTokenizer(),
        prompting=PromptingConfig(),
        prompt_input=PromptInput(
            question="Who wrote Hamlet?",
            evidence="Title: Hamlet\n- William Shakespeare wrote Hamlet.",
        ),
    )

    assert used_chat_template is False
    assert "System:" in rendered
    assert "Assistant:" in rendered


def test_postprocess_generated_text_strips_only_obvious_prefixes() -> None:
    assert postprocess_generated_text("Assistant: Answer: William Shakespeare\n") == "William Shakespeare"
