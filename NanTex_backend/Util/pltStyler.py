## Dependiencies
import matplotlib.pyplot as plt

# params
style: str = "seaborn-v0_8-bright"
dark_style: str = "dark_background"
font = {"family": "DejaVu Sans", "weight": "bold", "size": 12}


class pltStyler:
    style: str
    font: dict[str]

    def __init__(self, style: str = dark_style, font: dict[str] = font) -> None:
        self.style = style
        self.font = font

    def default_stylesheet(self) -> None:
        plt.style.use("default")

    def enforece_stylesheet(self) -> None:
        # just to have a clean slate
        self.default_stylesheet()

        # enforce the stylesheet
        plt.style.use(self.style)
        plt.rc("font", **self.font)
