import marimo

__generated_with = "0.11.13"
app = marimo.App(
    width="full",
    app_title="Analysis of Algae Biofuel Data",
    css_file="/home/abdullahalazmi/.local/share/mtheme/themes/default.css",
    auto_download=["ipynb", "html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="").center
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <div class="text-center">
            <h2>Development and Optimization of Algae-Based Biofuel Systems for Carbon-Negative Energy Production</h2>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <div class="text-center">
            <h2>Analysis of Algae Biofuel Data</h2>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Formulae to Analyses the Data

        <div class="">
            <ol type="1">
                <li>Biomass Productivity Analysis</li>
                <ul>
                    <li>Biomass Productivity</li>
                </ul>
                <li>Lipid Yield and Extraction Efficiency</li>
                <li>Carbon Sequestration Analysis</li>
                <li>Life Cycle Assesment</li>
            </ol>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. Biomass Productivity Analysis

        ### Specific Growth Rate

        \[
        \mu = \frac{\ln(x_t) - \ln(x_o)}{t}
        \]

        #### Where:

        - \( x_o \) = Initial Biomass Concentration (g/L)
        - \( x_t \) = Final Biomass Concentration (g/L)
        - \( t \) = Time (days)

        ## 1.1 Biomass Productivity

        \[
        P_b = \frac{x_t - x_o}{t}
        \]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Lipid Yield and Extraction Efficiency

        ### Lipid Context

        \[
        L_c = \frac{M_L}{M_B} \times 100
        \]

        #### Where:

        - \( M_L \) = Mass of extracted lipids (g)
        - \( M_B \) = Mass of dired biomass (g)

        ### Lipid Productivity

        \[
        P_C = P_B \times \frac{L_C}{100}
        \]
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Carbon Sequestration Analysis

        ### Carbon Fixation Efficiency

        \[
        CEE = \frac{C*B}{C*{CO_2}} \times 100
        \]

        #### Where:

        - \( C_B \) = Carbon content in algal biomass (g)
        - \( C_{CO_2} \) = Carbon dioxide supplied to the systems (g)

        ### \( C_{CO_2} \) Sequestration per Biomass

        \[
        S*{CO_2} = \frac{{M_B}\times{F_C}} {M*{CO_2}}
        \]

        #### Where:

        - \( F_C \) = Fraction of Carbon in Algal Biomass (g)
        - \( M_{CO_2} \) = Molecular weight of Carbon Dioxide (g/mol)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
