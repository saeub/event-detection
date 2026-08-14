import matplotlib.patches
import matplotlib.pyplot as plt
import polars as pl
import pymovements as pm
import streamlit as st

FIXATION_ALGORITHMS = {
    "idt": "I-DT",
    "ivt": "I-VT",
}
VELOCITY_METHODS = {
    "savitzky_golay": "Savitzky-Golay",
    "fivepoint": "Five-point",
    "neighbors": "Subsequent - preceding",
    "preceding": "Current - preceding",
}


@st.cache_data(max_entries=1)
def load_data(
    file,
    experiment_kwargs,
    velocity_kwargs,
):
    experiment_kwargs = dict(experiment_kwargs)
    velocity_kwargs = dict(velocity_kwargs)
    experiment = pm.Experiment(**experiment_kwargs)
    gaze = pm.gaze.from_csv(
        file, pixel_columns=["pixel_x", "pixel_y"], experiment=experiment
    )
    gaze.pix2deg()
    gaze.pos2vel(**velocity_kwargs)
    gaze.samples = gaze.samples.with_columns(
        _velocity=(
            pl.col("velocity").list.__getitem__(0).pow(2)
            + pl.col("velocity").list.__getitem__(1).pow(2)
        ).sqrt()
    )
    return gaze


data_choice = st.sidebar.selectbox(
    "Data",
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
else:
    file = st.sidebar.file_uploader("Samples file", type=["csv"])
    experiment_kwargs = {
        "sampling_rate": st.sidebar.number_input("Sampling rate (Hz)", value=1000),
        "screen_width_px": st.sidebar.number_input("Screen width (px)", value=1920),
        "screen_height_px": st.sidebar.number_input("Screen height (px)", value=1080),
        "screen_width_cm": st.sidebar.number_input("Screen width (cm)", value=52.7),
        "screen_height_cm": st.sidebar.number_input("Screen height (cm)", value=29.6),
        "distance_cm": st.sidebar.number_input("Eye-to-screen distance (cm)", value=60),
    }

fixation_tab, velocity_tab = st.tabs(["Fixation detection", "Velocity calculation"])
with fixation_tab:
    fixation_algorithm = st.selectbox(
        "Algorithm", FIXATION_ALGORITHMS.keys(), format_func=FIXATION_ALGORITHMS.get
    )
    if fixation_algorithm == "idt":
        fixation_kwargs = {
            "minimum_duration": st.slider("Minimum duration", 1, 1000, 100),
            "dispersion_threshold": st.slider("Dispersion threshold", 0.1, 2.0, 1.0),
        }
    elif fixation_algorithm == "ivt":
        fixation_kwargs = {
            "minimum_duration": st.slider("Minimum duration", 1, 1000, 100),
            "velocity_threshold": st.slider("Velocity threshold", 1.0, 80.0, 20.0),
        }
with velocity_tab:
    velocity_method = st.selectbox(
        "Method",
        ["savitzky_golay", "fivepoint", "neighbors", "preceding"],
        format_func=VELOCITY_METHODS.get,
    )
    if velocity_method == "savitzky_golay":
        velocity_kwargs = {
            "window_length": st.slider("Window length", 3, 51, 21, step=2),
            "degree": st.slider("Polynomial order", 1, 5, 2),
        }
    else:
        velocity_kwargs = {}
    velocity_kwargs["method"] = velocity_method

if file is not None:
    gaze = load_data(
        file,
        tuple(sorted(experiment_kwargs.items())),
        tuple(sorted(velocity_kwargs.items())),
    )
    min_time = gaze.samples["time"].min()
    max_time = gaze.samples["time"].max()
    col1, col2 = st.columns(2)
    duration = col2.slider("Duration (ms)", 1, 20000, 2000)
    start_time = col1.slider("Start (ms)", min_time, max_time - duration, min_time)
    gaze.samples = gaze.samples.filter(
        gaze.samples["time"].is_between(start_time, start_time + duration)
    )
    if gaze.samples.is_empty():
        st.warning("No samples in the selected time range.")
    else:
        gaze.detect(fixation_algorithm, **fixation_kwargs)
        gaze.compute_event_properties(("location", {"position_column": "pixel"}))
        gaze.unnest()
        gaze.events.unnest()

        fig, (x_ax, y_ax, vel_ax) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        x_ax.plot(gaze.samples["time"], gaze.samples["pixel_x"])
        x_ax.set_ylim(0, experiment_kwargs["screen_width_px"])
        x_ax.set_ylabel("X (px)")
        y_ax.plot(gaze.samples["time"], gaze.samples["pixel_y"])
        y_ax.set_ylim(0, experiment_kwargs["screen_height_px"])
        y_ax.set_ylabel("Y (px)")
        vel_ax.plot(
            gaze.samples["time"],
            gaze.samples["_velocity"],
        )
        vel_ax.set_ylabel("Velocity (deg/s)")
        vel_ax.set_xlabel("Time (ms)")
        for row in gaze.events.frame.iter_rows(named=True):
            x_ax.axvspan(row["onset"], row["offset"], color="red", alpha=0.3)
            y_ax.axvspan(row["onset"], row["offset"], color="red", alpha=0.3)
            vel_ax.axvspan(row["onset"], row["offset"], color="red", alpha=0.3)
        fig

        fig, ax = plt.subplots(
            figsize=(
                10,
                10
                * experiment_kwargs["screen_height_px"]
                / experiment_kwargs["screen_width_px"],
            )
        )
        ax.plot(gaze.samples["pixel_x"], gaze.samples["pixel_y"])
        ax.set_xlim(0, experiment_kwargs["screen_width_px"])
        ax.set_ylim(experiment_kwargs["screen_height_px"], 0)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        for row in gaze.events.frame.filter(pl.col("name") == "fixation").iter_rows(
            named=True
        ):
            ax.add_artist(
                matplotlib.patches.Circle(
                    (row["location_x"], row["location_y"]),
                    row["duration"] * 0.3,
                    color="red",
                    alpha=0.3,
                    zorder=10,
                )
            )
        fig

        code = "import pymovements as pm\n\n"
        code += f"experiment = pm.Experiment(\n"
        for k, v in experiment_kwargs.items():
            code += f"    {k}={v},\n"
        code += ")\n"
        code += f"# Load gaze data\n"
        code += f"gaze = pm.gaze.from_csv(\n"
        code += f"    {file.name!r}, pixel_columns=['pixel_x', 'pixel_y'], experiment=experiment\n"
        code += ")\n"
        code += f"# Convert pixel coordinates to degrees of visual angle\n"
        code += "gaze.pix2deg()\n"
        if fixation_algorithm == "ivt":
            code += f"# Convert gaze positions to velocities\n"
            pos2vel_kwargs = ", ".join(f"{k}={v!r}" for k, v in velocity_kwargs.items())
            code += f"gaze.pos2vel({pos2vel_kwargs})\n"
        code += f"# Detect fixations\n"
        detect_kwargs = ", ".join(f"{k}={v!r}" for k, v in fixation_kwargs.items())
        code += f"gaze.detect({fixation_algorithm!r}, {detect_kwargs})\n"
        code += f"# Compute fixation location\n"
        code += "gaze.compute_event_properties(('location', {'position_column': 'pixel'}))\n"
        st.code(code, language="python")
        st.download_button(
            "Download Python script",
            code,
            file_name="detect_fixations.py",
        )
