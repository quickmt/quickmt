from pathlib import Path

import monsterui.all as mui
from fasthtml.common import *
from fire import Fire
from monsterui.all import *
from quickmt import Translator
from quickmt.hub import hf_list, hf_download


hdrs = Theme.green.headers(mode="dark", highlightjs=True)

app, rt = fast_app(hdrs=hdrs)
app.title = "quickmt"

t = None


def translation_form(
    input_text: str = "",
    beam_size: int = 2,
    num_threads: int = 4,
    selected_model: str = None,
    selected_device: str = None,
    model_folder: str = str(Path(".").absolute()),
):
    quickmt_models = [i.split("/")[1] for i in hf_list()]
    return Form(
        Grid(
            LabelInput(
                "Quickmt Model Folder",
                name="model_folder",
                value=model_folder,
                data_uk_tooltip="title:Folder containing (or to save) quickmt models; pos:top-left",
            ),
            LabelSelect(
                *Options("Select a model", *quickmt_models, selected_idx=0, disabled_idxs={0}),
                label="Model",
                name="selected_model",
                value=selected_model,
            ),
        ),
        Grid(
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
        LabelTextArea("Input Text", name="input_text", style="min-height: 460px;"),
        Button("Translate", cls=(ButtonT.primary, "mt-2")),
        method="post",
        hx_target="#output_text",
        hx_post="/translate",
        hx_swap="innerHTML",
    )


def menu_bar():
    return NavBar(
        Ul(
            Li(
                A(
                    UkIcon("github", height=40, width=40),
                    href="https://github.com/quickmt/quickmt",
                    target="_blank",
                ),
            ),
        ),
        brand=A(H1("quickmt"), href="/"),
    )


@rt
def index():
    body_content = Grid(
        Div(translation_form(), cls="mt-5 mr-5"),
        Div(cols_lg=2, cls="mt-10 ml-5 mr-10", id="output_text"),
        cls="ml-10 mr-5",
    )
    return Title("quickmt"), Main(menu_bar(), body_content)


@rt
@app.post("/download_model")
def download_model(selected_model: str, model_folder: str):
    hf_download(model_name="quickmt/" + selected_model, output_dir=Path(model_folder) / selected_model)
    return Alert(
        DivLAligned(UkIcon("thumbsup"), B(f"Download complete"), P(f"Finished downloading {selected_model}")),
        cls=AlertT.success,
    )


def model_download_panel(selected_model: str, model_folder: str):
    return Alert(
        DivLAligned(UkIcon("triangle-alert"), B(f"Model not found"), P("Model must be downloaded first.")),
        cls=AlertT.error,
    ), Form(
        LabelInput("", name="model_folder", value=model_folder, hidden=True),
        LabelInput(
            "",
            hidden=True,
            name="selected_model",
            value=selected_model,
        ),
        Button("Download Model", cls=(ButtonT.primary, "mt-2")),
        method="post",
        hx_post="/download_model",
        hx_target="#output_text",
    )


@rt
@app.post("/translate")
def translate(
    input_text: str,
    beam_size: int,
    selected_model: str,
    selected_device: str,
    num_threads: int,
    model_folder: str,
):
    global t
    model_path = Path(model_folder) / selected_model

    if not model_path.exists():
        return model_download_panel(selected_model, model_folder)

    if t is None or str(selected_model) != str(t.model_path):
        try:
            t = Translator(str(model_path), device=selected_device, inter_threads=int(num_threads))
        except:
            if selected_device in ("auto", "gpu"):
                return Alert(
                    DivLAligned(
                        UkIcon("triangle-alert"),
                        B(f"Could not load model"),
                        P(
                            "Be sure the model is fully downloaded, the cuda toolkit is installed and you have enough available GPU RAM."
                        ),
                    ),
                    cls=AlertT.error,
                )

    return LabelTextArea(
        "Translation",
        id="output_text",
        style="min-height: 600px;",
        value="\n".join(t(input_text.splitlines(), beam_size=beam_size)),
    )


def main(host: str = "127.0.0.1", port: str = 5001, reload: bool = False):
    serve(host=host, port=port, reload=reload)


Fire(main)
