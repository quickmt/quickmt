from pathlib import Path

import uvicorn
from faicons import icon_svg as icon
from fire import Fire
from shiny import App, reactive, render, ui

from quickmt import Translator
from quickmt.hub import hf_download, hf_list

t = None

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.layout_columns(ui.input_dark_mode(), ui.p("Toggle Dark Mode")),
        ui.input_selectize(
            "model", "Select model", choices=[i.split("/")[1] for i in hf_list()]
        ),
        ui.input_text("model_folder", "Model Folder", value=str(Path(".").absolute())),
        ui.input_numeric("beam_size", "Beam size", min=1, max=10, step=1, value=2),
        ui.input_numeric("num_threads", "CPU Threads", min=1, max=16, step=1, value=4),
        ui.input_selectize(
            "compute_device",
            "Compute Device",
            choices=["auto", "cpu", "gpu"],
            selected="cpu",
        ),
        width="350px",
    ),
    ui.layout_columns(
        ui.card(
            ui.h5("Input Text"),
            ui.input_text_area(
                "input_text",
                "",
                value="",
                width="100%",
                height="600px",
            ),
            ui.input_action_button(
                "translate_button", "Translate!", class_="btn-primary"
            ),
        ),
        ui.card(ui.h5("Translation"), ui.output_ui("translate")),
    ),
    title="quickmt",
    theme=ui.Theme("flatly"),
)


def server(input, output, session):
    @render.ui
    @reactive.event(input.quickmt_model_download)  # Take a dependency on the button
    def model_download_output():
        print("DOWNLOADING")
        hf_download(
            model_name="quickmt/" + input.model(),
            output_dir=Path(input.model_folder()) / input.model(),
        )
        return "Model downloaded"

    @render.ui
    @reactive.event(input.translate_button)  # Take a dependency on the button
    def translate():
        global t
        model_path = Path(input.model_folder()) / input.model()
        try:
            if t is None or str(input.model()) != str(Path(t.model_path).name):
                print(f"Loading model {input.model()}")
                t = Translator(
                    str(model_path),
                    device=input.compute_device(),
                    inter_threads=int(input.num_threads()),
                )
            if len(input.input_text()) == 0:
                return ""

            return [
                ui.p(i)
                for i in t(input.input_text().splitlines(), beam_size=input.beam_size())
            ]

        except:
            return [
                ui.value_box(
                    title=f"Ensure model is downloaded to {input.model_folder()}",
                    value="Failed to load model",
                    showcase=icon("bug"),
                ),
                ui.input_action_button(
                    "quickmt_model_download", "Download Model", class_="btn-primary"
                ),
                ui.output_ui("model_download_output"),
            ]


def runapp(port: int = 8000, host: str = "127.0.0.1"):
    app = App(app_ui, server)
    uvicorn.run(app, port=port, host=host)

def main(): Fire(runapp)
    

if __name__ == "__main__":
    main()
