from pathlib import Path

import monsterui.all as mui
from fasthtml.common import *
from fire import Fire
from monsterui.all import *

from quickmt import Translator

GITHUB_SVG = """<svg width="40px" height="40px" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 40.172 33.405 46.69 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-9.127-13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17-4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052 4.367 7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6-10.839-1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2-.485-1.222-2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052a46.97 46.97 0 0 1 12.214-1.63c4.125 0 8.33.571 12.213 1.63 9.302-6.356 13.427-5.052 13.427-5.052 2.67 6.763.97 11.816.485 13.038 3.155 3.422 5.015 7.822 5.015 13.2 0 18.905-11.404 23.06-22.324 24.283 1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-.08 11.897-.08 13.526 0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-24.935 33.405-46.691C97.707 22 75.788 0 48.854 0z" fill="#fff"/></svg>"""
model_folder = Path(".")
models = [i for i in model_folder.glob("quickmt-*")]

t = None

# Get frankenui and tailwind headers via CDN using Theme.blue.headers()
hdrs = Theme.green.headers(mode="dark")
app, rt = fast_app(hdrs=hdrs, title="Test")


def translation_form(
    input_text: str = "",
    beam_size: int = 2,
    num_threads: int = 4,
    selected_model: str = None,
    selected_device: str = None,
):
    return Form(
        Grid(
            LabelSelect(
                *Options("Select a model", *models, selected_idx=0, disabled_idxs={0}),
                label="Model",
                name="selected_model",
                value=selected_model,
            ),
            LabelSelect(
                *Options("Compute Device", *("auto", "cpu", "cuda"), selected_idx=1, disabled_idxs={0}),
                label="Device",
                name="selected_device",
                value=selected_device,
                data_uk_tooltip="title:Auto will use GPU if available and CPU otherwise; pos:top-left",
            ),
            LabelInput(
                "Beam Size",
                name="beam_size",
                value=beam_size,
                data_uk_tooltip="title:Balances speed and quality. Set to 1 for fastest speed, 8 for higher quality; pos:top-left",
            ),
            LabelInput(
                "CPU Cores",
                name="num_threads",
                value=num_threads,
                data_uk_tooltip="title:Number of CPU cores to use for multi-threading; pos:top-left",
            ),
            cls="space-x-2",
        ),
        LabelTextArea("Input Text", name="input_text", style="min-height: 540px;"),
        Button("Translate", cls=(ButtonT.primary, "mt-2")),
        method="post",
        hx_target="#output_text",
        hx_post="/translate",
        hx_swap="innerHTML",
    )


def error_message(body: str, title: str):
    return Alert(DivLAligned(UkIcon("triangle-alert"), B(f"ERROR: {title}"), P(body)), cls=AlertT.error)


def menu_bar():
    return NavBar(
        Ul(
            Li(
                A(
                    NotStr(GITHUB_SVG),
                    href="https://github.com/quickmt/quickmt",
                    target="_blank",
                ),
            ),
        ),
        brand=A(H1("quickmt"), href="/"),
    )


@rt
def index():
    if len(models) > 0:
        body_content = Grid(
            Div(translation_form(), cls="mt-5 mr-5"),
            Div(cols_lg=2, cls="mt-10 ml-5 mr-10", id="output_text"),
            cls="ml-10 mr-5",
        )

    else:
        body_content = error_message(
            title="quickmt models not found",
            body="No quickmt models were found in current working directory. Please change directories before launching the quickmt-app or download models into this folder.",
        )
    return Title("quickmt"), Main(menu_bar(), body_content)


@rt
@app.post("/translate")
def translate(input_text: str, beam_size: int, selected_model: str, selected_device: str, num_threads: int):
    global t
    if t is None or str(selected_model) != str(t.model_path):
        print(f"Loading model {selected_model}")
        try:
            t = Translator(
                str(Path(model_folder) / selected_model), device=selected_device, inter_threads=int(num_threads)
            )
        except:
            if selected_device in ("auto", "gpu"):
                return error_message(
                    title="Could not load model",
                    body="Be sure the cuda toolkit is installed and you have enough available GPU RAM.",
                )

    return LabelTextArea(
        "Translation",
        id="output_text",
        style="min-height: 600px;",
        value="\n".join(t(input_text.splitlines(), beam_size=beam_size)),
    )


def main(host: str = "127.0.0.1", port: str = 5001, reload: bool = True):
    serve(host=host, port=port, reload=reload)


Fire(main)
