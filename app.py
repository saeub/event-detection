import matplotlib.patches
import matplotlib.pyplot as plt
import polars as pl
import pymovements as pm
import streamlit as st

FIXATION_ALGORITHMS = {
    None: "None",
    "idt": "I-DT",
    "ivt": "I-VT",
}
SACCADE_ALGORITHMS = {
    None: "None",
    "microsaccades": "Microsaccade",
}
MICROSACCADE_THRESHOLDS = {
    "std": "Standard deviation",
    "mad": "Median absolute deviation",
    "engbert2003": "Engbert & Kliegl (2003)",
    "engbert2015": "Engbert (2015)",
}
VELOCITY_METHODS = {
    "savitzky_golay": "Savitzky-Golay",
    "fivepoint": "Five-point",
    "neighbors": "Subsequent - preceding",
    "preceding": "Current - preceding",
}


st.set_page_config(
    page_title="Eye movement event detection",
    page_icon="👁️",
)
st.title("Eye movement event detection")


@st.cache_data(max_entries=1)
def load_data(
    file,
    experiment_kwargs,
    velocity_kwargs,
    trial_columns,
):
    experiment_kwargs = dict(experiment_kwargs)
    velocity_kwargs = dict(velocity_kwargs)
    experiment = pm.Experiment(**experiment_kwargs)
    if file.name.endswith(".csv"):
        gaze = pm.gaze.from_csv(
            file,
            pixel_columns=["pixel_x", "pixel_y"],
            experiment=experiment,
            trial_columns=trial_columns,
        )
    elif file.name.endswith(".tsv"):
        gaze = pm.gaze.from_csv(
            file,
            pixel_columns=["pixel_x", "pixel_y"],
            experiment=experiment,
            trial_columns=trial_columns,
            read_csv_kwargs={"separator": "\t"},
        )
    elif file.name.endswith(".asc"):
        gaze = pm.gaze.from_asc(
            file,
            experiment=experiment,
            patterns=[
                r"TRIALID (?P<trial_id>.+)",
                {
                    "pattern": r"TRIAL_RESULT .+",
                    "column": "trial_id",
                    "value": None,
                },
            ],
            trial_columns=trial_columns,
        )
    else:
        raise ValueError(f"Unsupported file type: {file.name}")
    gaze.pix2deg()
    gaze.pos2vel(**velocity_kwargs)
    gaze.samples = gaze.samples.with_columns(
        _velocity=(
            pl.col("velocity").list.__getitem__(0).pow(2)
            + pl.col("velocity").list.__getitem__(1).pow(2)
        ).sqrt()
    )
    return gaze


st.sidebar.title("Data settings")

data_choice = st.sidebar.selectbox(
    "Gaze data",
    ["Example data (GGTG)", "Upload your own"],
)
if data_choice == "Example data (GGTG)":
    file = open("examples/ggtg.P01.csv", "rb")
    experiment_kwargs = {
        "sampling_rate": 1000,
        "screen_width_px": 1100,
        "screen_height_px": 900,
        "screen_width_cm": 31.2,
        "screen_height_cm": 25.2,
        "distance_cm": 66,
    }
    trial_columns = ["stimulus"]
else:
    file = st.sidebar.file_uploader("Gaze file", type=["csv", "tsv", "asc"])
    experiment_kwargs = {
        "sampling_rate": st.sidebar.number_input("Sampling rate (Hz)", value=1000),
        "screen_width_px": st.sidebar.number_input(
            "Screen width (px)", value=1920, key="screen_width_px"
        ),
        "screen_height_px": st.sidebar.number_input(
            "Screen height (px)", value=1080, key="screen_height_px"
        ),
        "screen_width_cm": st.sidebar.number_input("Screen width (cm)", value=53.0),
        "screen_height_cm": st.sidebar.number_input("Screen height (cm)", value=30.0),
        "distance_cm": st.sidebar.number_input(
            "Eye-to-screen distance (cm)", value=60.0
        ),
    }
    if file is not None:
        if file.name.endswith(".asc"):
            trial_columns = ["trial_id"]
        else:
            trial_columns = st.sidebar.text_input(
                "Trial columns (comma-separated)", value=None
            )
            if trial_columns:
                trial_columns = [column.strip() for column in trial_columns.split(",")]
            else:
                trial_columns = None

fixation_tab, saccade_tab, velocity_tab = st.tabs(
    ["Fixation detection", "Saccade detection", "Velocity calculation"]
)
with fixation_tab:
    algorithm_choices = list(FIXATION_ALGORITHMS.keys())
    fixation_algorithm = st.selectbox(
        "Algorithm",
        algorithm_choices,
        format_func=FIXATION_ALGORITHMS.get,
        index=algorithm_choices.index("ivt"),
    )
    if fixation_algorithm == "idt":
        col1, col2 = st.columns(2)
        fixation_kwargs = {
            "dispersion_threshold": col1.slider("Dispersion threshold", 0.1, 2.0, 1.0),
            "minimum_duration": col2.slider("Minimum duration", 1, 1000, 100),
        }
    elif fixation_algorithm == "ivt":
        col1, col2 = st.columns(2)
        fixation_kwargs = {
            "minimum_duration": col2.slider("Minimum duration", 1, 1000, 100),
            "velocity_threshold": col1.slider("Velocity threshold", 1.0, 80.0, 20.0),
        }
with saccade_tab:
    algorithm_choices = list(SACCADE_ALGORITHMS.keys())
    saccade_algorithm = st.selectbox(
        "Algorithm",
        algorithm_choices,
        format_func=SACCADE_ALGORITHMS.get,
        index=algorithm_choices.index(None),
    )
    if saccade_algorithm == "microsaccades":
        threshold_choices = list(MICROSACCADE_THRESHOLDS.keys())
        col1, col2, col3 = st.columns(3)
        saccade_kwargs = {
            "threshold": col1.selectbox(
                "Threshold",
                threshold_choices,
                format_func=MICROSACCADE_THRESHOLDS.get,
                index=threshold_choices.index("engbert2015"),
            ),
            "threshold_factor": col2.slider("Threshold factor", 1.0, 20.0, 6.0),
            "minimum_duration": col3.slider("Minimum duration", 1, 1000, 6),
        }
with velocity_tab:
    velocity_method = st.selectbox(
        "Method",
        ["savitzky_golay", "fivepoint", "neighbors", "preceding"],
        format_func=VELOCITY_METHODS.get,
    )
    if velocity_method == "savitzky_golay":
        col1, col2 = st.columns(2)
        velocity_kwargs = {
            "window_length": col1.slider("Window length", 3, 51, 21, step=2),
            "degree": col2.slider("Polynomial order", 1, 5, 2),
        }
    else:
        velocity_kwargs = {}
    velocity_kwargs["method"] = velocity_method

if file is not None:
    st.subheader("Preview")

    gaze = load_data(
        file,
        tuple(sorted(experiment_kwargs.items())),
        tuple(sorted(velocity_kwargs.items())),
        trial_columns,
    )
    if trial_columns is not None:
        # combine trial column values with | as separator
        trials = (
            gaze.samples.select(trial_columns)
            .unique(maintain_order=True)
            .map_rows(lambda row: "|".join(map(str, row)))
            .to_series()
            .to_list()
        )
        trial = st.selectbox("Trial", trials)
        gaze.samples = gaze.samples.filter(
            pl.concat_list([pl.col(c) for c in trial_columns]).map_elements(
                lambda x: "|".join(map(str, x))
            )
            == trial
        )

    min_time = gaze.samples["time"].min()
    max_time = gaze.samples["time"].max()
    col1, col2 = st.columns(2)
    duration = col2.slider("Duration (ms)", 1, 50000, 2000)
    start_time = col1.slider("Start (ms)", min_time, max_time - duration, min_time)
    gaze.samples = gaze.samples.filter(
        gaze.samples["time"].is_between(start_time, start_time + duration)
    )
    if gaze.samples.is_empty():
        st.warning("No samples in the selected time range.")
    else:
        if fixation_algorithm is not None:
            gaze.detect(fixation_algorithm, **fixation_kwargs, name="fixation")
            gaze.compute_event_properties(
                ("location", {"position_column": "pixel"}),
                "mean_location",
                name="fixation",
            )
        if saccade_algorithm is not None:
            gaze.detect(saccade_algorithm, **saccade_kwargs, name="saccade")
            gaze.compute_event_properties("amplitude", name="saccade")
            gaze.compute_event_properties("peak_velocity", name="saccade")
            gaze.compute_event_properties(
                ("location", {"position_column": "pixel", "method": "first"}),
                "start_location",
                name="saccade",
            )
            gaze.compute_event_properties(
                ("location", {"position_column": "pixel", "method": "last"}),
                "end_location",
                name="saccade",
            )
        gaze.unnest()
        gaze.events.unnest()

        fixations = gaze.events.fixations
        saccades = gaze.events.saccades

        timeseries_tab, scanpath_tab, mainsequence_tab, code_tab = st.tabs(
            ["Time series", "Scanpath", "Main sequence", "Python code"]
        )

        with timeseries_tab:
            fig, (x_ax, y_ax, vel_ax) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
            x_ax.plot(gaze.samples["time"], gaze.samples["pixel_x"], color="gray")
            x_ax.set_ylim(0, experiment_kwargs["screen_width_px"])
            x_ax.set_ylabel("X (px)")
            y_ax.plot(gaze.samples["time"], gaze.samples["pixel_y"], color="gray")
            y_ax.set_ylim(0, experiment_kwargs["screen_height_px"])
            y_ax.set_ylabel("Y (px)")
            vel_ax.plot(gaze.samples["time"], gaze.samples["_velocity"], color="gray")
            vel_ax.set_ylabel("Velocity (deg/s)")
            vel_ax.set_xlabel("Time (ms)")
            for row in fixations.iter_rows(named=True):
                x_ax.axvspan(row["onset"], row["offset"], color="red", alpha=0.3)
                y_ax.axvspan(row["onset"], row["offset"], color="red", alpha=0.3)
                vel_ax.axvspan(row["onset"], row["offset"], color="red", alpha=0.3)
            for row in saccades.iter_rows(named=True):
                x_ax.axvspan(row["onset"], row["offset"], color="blue", alpha=0.3)
                y_ax.axvspan(row["onset"], row["offset"], color="blue", alpha=0.3)
                vel_ax.axvspan(row["onset"], row["offset"], color="blue", alpha=0.3)
            fig

        with scanpath_tab:
            fig, ax = plt.subplots(
                figsize=(
                    10,
                    10
                    * experiment_kwargs["screen_height_px"]
                    / experiment_kwargs["screen_width_px"],
                )
            )
            ax.plot(gaze.samples["pixel_x"], gaze.samples["pixel_y"], color="gray")
            ax.set_xlim(0, experiment_kwargs["screen_width_px"])
            ax.set_ylim(experiment_kwargs["screen_height_px"], 0)
            ax.set_xlabel("X (px)")
            ax.set_ylabel("Y (px)")
            for row in fixations.iter_rows(named=True):
                ax.add_artist(
                    matplotlib.patches.Circle(
                        (row["mean_location_x"], row["mean_location_y"]),
                        row["duration"] * 0.3,
                        color="red",
                        alpha=0.3,
                        zorder=10,
                    )
                )
            for row in saccades.iter_rows(named=True):
                ax.add_artist(
                    matplotlib.patches.FancyArrowPatch(
                        (row["start_location_x"], row["start_location_y"]),
                        (row["end_location_x"], row["end_location_y"]),
                        color="blue",
                        alpha=0.5,
                        zorder=10,
                        arrowstyle="->",
                        mutation_scale=20,
                        linewidth=2,
                    )
                )
            fig

        with mainsequence_tab:
            if saccade_algorithm is not None and not saccades.is_empty():
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(
                    saccades["amplitude"],
                    saccades["peak_velocity"],
                    color="blue",
                    alpha=0.5,
                )
                ax.set_xlabel("Amplitude (deg)")
                ax.set_ylabel("Peak velocity (deg/s)")
                fig
            else:
                st.info("No saccades detected. Main sequence plot cannot be generated.")

        with code_tab:
            code = "import pymovements as pm\n\n"
            code += f"experiment = pm.Experiment(\n"
            for k, v in experiment_kwargs.items():
                code += f"    {k}={v},\n"
            code += ")\n"
            code += f"# Load gaze data\n"
            if file.name.endswith(".csv"):
                code += f"gaze = pm.gaze.from_csv(\n"
                code += f"    {file.name!r}, pixel_columns=['pixel_x', 'pixel_y'], experiment=experiment\n"
                code += ")\n"
            elif file.name.endswith(".tsv"):
                code += f"gaze = pm.gaze.from_csv(\n"
                code += f"    {file.name!r}, pixel_columns=['pixel_x', 'pixel_y'], experiment=experiment, read_csv_kwargs={{'separator': '\\t'}}\n"
                code += ")\n"
            elif file.name.endswith(".asc"):
                code += f"gaze = pm.gaze.from_asc(\n"
                code += f"    {file.name!r}, experiment=experiment,\n"
                code += f"    patterns=[\n"
                code += f"        r'TRIALID (?P<trial_id>.+)',\n"
                code += f"        {{'pattern': r'TRIAL_RESULT .+', 'column': 'trial_id', 'value': None}},\n"
                code += f"    ],\n"
                code += f"    trial_columns={trial_columns!r},\n"
                code += ")\n"
            code += f"# Convert pixel coordinates to degrees of visual angle\n"
            code += "gaze.pix2deg()\n"
            if fixation_algorithm == "ivt":
                code += f"# Convert gaze positions to velocities\n"
                pos2vel_kwargs = ", ".join(
                    f"{k}={v!r}" for k, v in velocity_kwargs.items()
                )
                code += f"gaze.pos2vel({pos2vel_kwargs})\n"
            if fixation_algorithm is not None:
                code += f"# Detect fixations\n"
                detect_kwargs = ", ".join(
                    f"{k}={v!r}" for k, v in fixation_kwargs.items()
                )
                code += f"gaze.detect({fixation_algorithm!r}, {detect_kwargs})\n"
                code += "gaze.compute_event_properties(('location', {'position_column': 'pixel'}), name='fixation')\n"
            if saccade_algorithm is not None:
                code += f"# Detect saccades\n"
                detect_kwargs = ", ".join(
                    f"{k}={v!r}" for k, v in saccade_kwargs.items()
                )
                code += f"gaze.detect({saccade_algorithm!r}, {detect_kwargs})\n"
                code += "gaze.compute_event_properties('amplitude', name='saccade')\n"
                code += (
                    "gaze.compute_event_properties('peak_velocity', name='saccade')\n"
                )
            st.code(code, language="python")
            st.download_button(
                "Download Python script",
                code,
                file_name="detect_fixations.py",
            )

st.bottom.html("""
<div style="text-align: center; font-size: 0.8em; color: gray;">
    Made by <a href="https://saeub.github.io" target="_blank">saeub</a>.
    Powered by <a href="https://pymovements.readthedocs.io/" target="_blank">pymovements</a>
    and <a href="https://streamlit.io/" target="_blank">Streamlit</a>.
</div>
""")
