from pathlib import Path

from fasthtml.common import *
from fire import Fire

# MonsterUI shadows fasthtml components with the same name
from monsterui.all import *

from quickmt import Translator

model_folder = Path(".")
models = [i for i in model_folder.glob("quickmt-*")]
t = Translator(str(Path(model_folder) / models[0]))

# Get frankenui and tailwind headers via CDN using Theme.blue.headers()
hdrs = Theme.blue.headers()

# fast_app is shadowed by MonsterUI to make it default to no Pico, and add body classes
# needed for frankenui theme styling
app, rt = fast_app(hdrs=hdrs)


def translation_form(input_text: str = "", beam_size: int = 5, selected_model: str = None):
    return Form(
        Group(
            LabelSelect(
                *Options("Select a model", *models, selected_idx=1, disabled_idxs={0}),
                label="Model",
                name="selected_model",
                id="selected_model",
                value=selected_model
            ),
            LabelInput("Beam Size", name="beam_size", value=beam_size),
            LabelTextArea(
                "Input Text", name="input_text", value=input_text, placeholder="", style="min-height: 400px;"
            ),
            Button("Translate", cls=(ButtonT.primary, "mt-2")),
        ),
        method="post",
        hx_target="#output_text",
        hx_post="/translate",
    )


@rt
def index():
    # Titled using a H1 title, sets the page title, and wraps contents in Main(Container(...)) using
    # frankenui styles. Generally you will want to use Titled for all of your pages
    return Titled(
        "quickmt",
        Grid(
            Div(translation_form(), style="margin-top: 20px;"),
            Div(
                Form(
                    LabelTextArea("Output Text", id="output_text", placeholder="", style="min-height: 600px;"),
                ),
                cols_lg=2,
                style="margin-top: 20px;",
            ),
        ),
    )


@rt
@app.post("/translate")
def translate(input_text: str, beam_size: int, selected_model: str):
    global t
    if str(selected_model) != str(t.model_path):
        print(f"Loading model {selected_model}")
        t = Translator(str(Path(model_folder) / selected_model))

    return "\n".join(t(input_text.splitlines(), beam_size=beam_size))


def main(host: str = "127.0.0.1", port: str = 5001):
    serve(host=host, port=port)


Fire(main)
