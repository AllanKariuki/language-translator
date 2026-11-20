"""Gradio web demo for English <-> Swahili translation.

Usage:
    python app.py [--model PATH] [--port 7860] [--share]
"""
import argparse
import os

import gradio as gr

from engine import Translator
from feedback import FeedbackStore

DIRECTIONS = {
    "English → Swahili": ("eng_Latn", "swh_Latn"),
    "Swahili → English": ("swh_Latn", "eng_Latn"),
}


def default_model_path():
    ct2 = "models/nllb-en-sw-ct2"
    return ct2 if os.path.exists(ct2) else "models/nllb-en-sw"


def build_demo(model_path):
    translator = Translator(model_path)
    store = FeedbackStore("feedback.db")

    def translate(text, direction):
        src, tgt = DIRECTIONS[direction]
        try:
            return translator.translate(text, src, tgt)
        except ValueError as e:
            raise gr.Error(str(e))

    def save_feedback(direction, source, output, rating=None, correction=None):
        if not output:
            return "Translate something first."
        src, tgt = DIRECTIONS[direction]
        try:
            store.add(src, tgt, source, output, rating=rating, correction=correction)
            return "Thanks — feedback saved."
        except ValueError as e:
            return str(e)

    def rate(direction, source, output, rating):
        return save_feedback(direction, source, output, rating=rating)

    def correct(direction, source, output, suggestion):
        return save_feedback(direction, source, output, correction=suggestion)

    with gr.Blocks() as demo:
        gr.Markdown("# English ⇄ Swahili translator")
        gr.Markdown("Fine-tuned NLLB-200 for English–Swahili translation.")

        direction = gr.Radio(list(DIRECTIONS), value="English → Swahili", label="Direction")
        with gr.Row():
            source = gr.Textbox(lines=5, placeholder="Type or paste text to translate ...", label="Input")
            output = gr.Textbox(lines=5, interactive=False, label="Translation")
        with gr.Row():
            translate_btn = gr.Button("Translate")
            swap_btn = gr.Button("⇄ Swap")

        translate_btn.click(translate, inputs=[source, direction], outputs=output, api_name="translate")

        def swap(direction, source, output):
            new = "Swahili → English" if direction == "English → Swahili" else "English → Swahili"
            return new, output, source

        swap_btn.click(swap, inputs=[direction, source, output], outputs=[direction, source, output])

        with gr.Row():
            up_btn = gr.Button("👍")
            down_btn = gr.Button("👎")
        with gr.Row():
            suggestion = gr.Textbox(lines=2, placeholder="Suggest a better translation", label="Suggestion")
            correct_btn = gr.Button("Submit correction")
        status = gr.Markdown()

        up_btn.click(
            lambda d, s, o: rate(d, s, o, "up"),
            inputs=[direction, source, output],
            outputs=status,
            api_name="rate",
        )
        down_btn.click(
            lambda d, s, o: rate(d, s, o, "down"),
            inputs=[direction, source, output],
            outputs=status,
        )
        correct_btn.click(
            correct,
            inputs=[direction, source, output, suggestion],
            outputs=status,
            api_name="correct",
        )

        gr.Markdown("Ratings and corrections are saved anonymously and may be used to improve the model.")

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="model dir (default: CT2 dir if present, else models/nllb-en-sw)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    model_path = args.model or default_model_path()
    demo = build_demo(model_path)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
