import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # Essential for Matplotlib to work with Tkinter GUI
import tifffile
import cv2
from scipy.spatial import distance_matrix
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import os
import sys

# --- Global variable to hold the main Tkinter root ---

def on_closing():
    if messagebox.askokcancel("Quit", "Are you sure you want to quit? No data will be exported if you quit now."):
        root.destroy()  # or your main window variable
        sys.exit()      # clean exit
main_tk_root = None

def browse_file():
    """
    Initiates the file Browse dialog and starts the tracking process.
    Manages the single main Tkinter root window.
    """
    global main_tk_root
    main_tk_root = tk.Tk()
    main_tk_root.protocol("WM_DELETE_WINDOW", on_closing)

    main_tk_root.withdraw() # Hide the main root window

    file_path = filedialog.askopenfilename(
        title="Select TIFF File for Larva Tracking",
        filetypes=[("TIFF files", "*.tif *.tiff")],
        parent=main_tk_root # Parent the file dialog to the main root
    )
    if file_path:
        track_larvae(file_path)
    else:
        print("No file selected. Exiting.")
        main_tk_root.destroy() # Destroy root if no file is selected
        return # Exit to prevent mainloop from starting unnecessarily
    
    main_tk_root.mainloop() # Start the Tkinter event loop for the main root

def track_larvae(tif_path):
    """
    Core function for tracking larvae across frames.
    Handles user input for analysis parameters and displays a progress bar.
    Filters out larvae that are not consistently available for the selected period.
    """
    max_match_distance = 50 # Maximum distance (pixels) for a larva to be considered a match

    try:
        frames = tifffile.imread(tif_path)
    except Exception as e:
        messagebox.showerror("File Error", f"Could not read TIFF file: {e}")
        return

    total_frames = len(frames)
    if total_frames == 0:
        messagebox.showerror("File Error", "The selected TIFF file contains no frames.")
        return

    # --- User Inputs for Tracking Parameters ---
    num_frames = simpledialog.askinteger(
        "Frames to Analyze",
        f"Total frames: {total_frames}\nEnter number of frames to analyze:",
        minvalue=1, maxvalue=total_frames, parent=main_tk_root
    )
    if not num_frames: # User cancelled frame selection
        messagebox.showinfo("Cancelled", "Frame count not provided. Larva tracking process aborted.")
        return # Exit here, as tracking is impossible without frame count

    frame_rate_fps = simpledialog.askfloat(
        "Frame Rate", "Enter video frame rate (FPS):",
        minvalue=0.1, parent=main_tk_root
    )
    if frame_rate_fps is None: # User cancelled or entered invalid
        messagebox.showwarning("Input Missing", "Frame rate not provided. Velocity calculation (per second) will be skipped.")
        
    conversion_factor_mm_per_px = simpledialog.askfloat(
        "Conversion Factor", "Enter conversion factor (e.g., mm per pixel):",
        minvalue=0.0001, parent=main_tk_root
    )
    if conversion_factor_mm_per_px is None: # User cancelled or entered invalid
        messagebox.showwarning("Input Missing", "Conversion factor not provided. Velocity (mm/s) will be skipped.")

    frames_to_process = frames[:num_frames]
    larva_tracks = [] # Stores lists of (x, y) tuples for each larva

    # --- Initial Frame Processing (Frame 0) ---
    frame0 = frames_to_process[0]
    if frame0.dtype != np.uint8:
        frame0 = cv2.normalize(frame0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, thresh0 = cv2.threshold(frame0, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours0, _ = cv2.findContours(thresh0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours0:
        messagebox.showwarning("No Larvae Detected", "No larvae were detected in the first frame. Cannot proceed with tracking.")
        return

    for cnt in contours0:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            larva_tracks.append([(cx, cy)])

    # --- Progress Bar Setup ---
    progress_window = tk.Toplevel(main_tk_root)
    progress_window.title("Tracking Progress")
    progress_window.geometry("350x100")
    progress_window.resizable(False, False)
    
    main_tk_root.update_idletasks()
    x = main_tk_root.winfo_x() + main_tk_root.winfo_width() // 2 - progress_window.winfo_width() // 2
    y = main_tk_root.winfo_y() + main_tk_root.winfo_height() // 2 - progress_window.winfo_height() // 2
    progress_window.geometry(f"+{x}+{y}")
    
    ttk.Label(progress_window, text="Tracking larvae...").pack(padx=10, pady=5)
    progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate', maximum=num_frames - 1)
    progress_bar.pack(padx=10, pady=10)
    
    progress_window.grab_set()
    progress_window.transient(main_tk_root)
    progress_window.update()

    # --- Track Matching Across Frames ---
    for i in range(1, num_frames):
        frame = frames_to_process[i]
        if frame.dtype != np.uint8:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_centroids.append((cx, cy))

        previous_positions = [track[-1] for track in larva_tracks]
        
        if not previous_positions:
            # If all larvae were lost, and no new ones detected in current_centroids,
            # we can't continue tracking existing tracks.
            # Any remaining tracks in larva_tracks will be filled with NaNs up to num_frames later.
            break 

        if current_centroids:
            dist_mat = distance_matrix(previous_positions, current_centroids)

            matches = [-1] * len(previous_positions)
            current_centroid_matched = [False] * len(current_centroids)

            flat_indices = dist_mat.argsort(axis=None)

            for flat_idx in flat_indices:
                prev_idx = flat_idx // dist_mat.shape[1]
                curr_idx = flat_idx % dist_mat.shape[1]
                dist_val = dist_mat[prev_idx, curr_idx]

                if dist_val < max_match_distance and \
                   not current_centroid_matched[curr_idx] and \
                   matches[prev_idx] == -1:

                    larva_tracks[prev_idx].append(current_centroids[curr_idx])
                    matches[prev_idx] = curr_idx
                    current_centroid_matched[curr_idx] = True
            
            for prev_idx, matched_curr_idx in enumerate(matches):
                if matched_curr_idx == -1:
                    larva_tracks[prev_idx].append((np.nan, np.nan))

        else: # No larvae detected in the current frame at all
            for track in larva_tracks:
                track.append((np.nan, np.nan))

        progress_bar['value'] = i
        progress_window.update_idletasks()

    # After the loop, ensure all tracks are of 'num_frames' length
    # by padding with NaNs if tracking stopped early
    for track in larva_tracks:
        while len(track) < num_frames:
            track.append((np.nan, np.nan))

    progress_window.destroy()

    # --- Filtering for consistently available larvae ---
    consistently_available_tracks = []
    num_nan_filtered = 0
    for i, track in enumerate(larva_tracks):
        track_np = np.array(track)
        # Check if any NaN exists in the (x,y) coordinates within the num_frames period
        if not np.any(np.isnan(track_np)):
            consistently_available_tracks.append((i, track))
        else:
            num_nan_filtered += 1

    if not consistently_available_tracks:
        messagebox.showwarning("No Consistent Larvae", 
                               f"No larvae were consistently detected for all {num_frames} frames. "
                               "Consider analyzing fewer frames or adjusting detection parameters.")
        return

    messagebox.showinfo("Filtering Complete", 
                        f"Filtered out {num_nan_filtered} larva tracks due to intermittent detection. "
                        f"{len(consistently_available_tracks)} larvae remain for analysis.")

    launch_plot_and_selection_ui(
        consistently_available_tracks, num_frames, tif_path,
        frame_rate_fps, conversion_factor_mm_per_px
    )

def calculate_velocities(tracks, names, num_frames, fps, mm_per_px):
    """
    Calculates path length and average velocities (px/s and mm/s) for each track.
    """
    velocities_data = []
    # Total time of the analyzed video segment
    total_time_sec = num_frames / fps if fps is not None and fps > 0 else np.nan

    for name, track in zip(names, tracks):
        track_np = np.array(track)
        
        diffs = np.diff(track_np, axis=0) 
        dists = np.linalg.norm(diffs, axis=1)

        path_length_px = np.nansum(dists) # nansum is fine here, as filtered tracks should not have NaNs

        velocity_px_s = np.nan
        velocity_mm_s = np.nan

        if not np.isnan(total_time_sec) and total_time_sec > 0:
            velocity_px_s = path_length_px / total_time_sec
            if mm_per_px is not None:
                velocity_mm_s = velocity_px_s * mm_per_px

        velocities_data.append({
            "Larva": name,
            "Path Length (px)": round(path_length_px, 2),
            "Velocity (px/s)": round(velocity_px_s, 2) if not np.isnan(velocity_px_s) else 'N/A',
            "Velocity (mm/s)": round(velocity_mm_s, 2) if not np.isnan(velocity_mm_s) else 'N/A'
        })
    return velocities_data

def save_tracks_and_velocities(tracks, names, num_frames, tif_path, fps, mm_per_px, fig):
    """
    Saves larva path data, calculated velocities, and a plot of trajectories to CSV and PNG files.
    """
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    output_dir = os.path.dirname(tif_path)

    # --- Save Larva Paths CSV ---
    data = {'Frame': list(range(num_frames))}
    for track, name in zip(tracks, names):
        x_vals = [pos[0] for pos in track]
        y_vals = [pos[1] for pos in track]
        data[f'{name}_X'] = x_vals
        data[f'{name}_Y'] = y_vals
        
    df_paths = pd.DataFrame(data)
    out_csv_paths = os.path.join(output_dir, f"{base_name}_larva_paths.csv")
    df_paths.to_csv(out_csv_paths, index=False)
    print(f"Saved larva paths to: {out_csv_paths}")

    # --- Calculate and Save Velocities CSV ---
    if fps is not None and fps > 0: 
        velocities_data = calculate_velocities(
            tracks, names, num_frames, fps, mm_per_px
        )
        df_velocities = pd.DataFrame(velocities_data)
        out_csv_velocities = os.path.join(output_dir, f"{base_name}_velocities.csv")
        df_velocities.to_csv(out_csv_velocities, index=False)
        print(f"Saved velocities to: {out_csv_velocities}")
    else:
        print("Frame rate not provided or invalid (0), skipping velocity calculation and export.")

    # --- Save Plot ---
    out_plot = os.path.join(output_dir, f"{base_name}_trajectories.png")
    fig.savefig(out_plot, dpi=300)
    print(f"Saved plot to: {out_plot}")


def launch_plot_and_selection_ui(detected_tracks, num_frames, tif_path, fps, mm_per_px):
    """
    Launches the GUI for displaying trajectories and allowing user selection for export.
    """
    window = tk.Toplevel(main_tk_root) # Parent to main_tk_root
    window.title("Larva Tracker – Deselect Larvae to Exclude")
    window.geometry("1000x800")
    
    def on_plot_window_closing():
        if messagebox.askokcancel("Quit", "Are you sure you want to quit? No data will be exported if you quit now."):
            window.destroy()
            if main_tk_root:
                main_tk_root.quit()
                main_tk_root.destroy()

    window.protocol("WM_DELETE_WINDOW", on_plot_window_closing)


    fig, ax = plt.subplots(figsize=(7, 7))
    plotted_lines = []
    colors = plt.cm.tab10.colors
    
    larva_names_in_ui = []
    check_vars = []

    for i, (original_index, track) in enumerate(detected_tracks):
        arr = np.array(track)
        line_color = colors[i % len(colors)]
        
        # All tracks here should be consistently available due to prior filtering,
        # so no need for 'if np.any(~np.isnan(arr))' checks here for plotting visibility
        line, = ax.plot(arr[:, 0], arr[:, 1], linestyle='-', marker='o', markersize=2,
                        label=f"Larva_{original_index+1}", color=line_color)
        
        # Add text label at the end
        ax.text(arr[-1, 0], arr[-1, 1], f"L{original_index+1}",
                fontsize=8, color=line_color, ha='left', va='center')
            
        plotted_lines.append(line)

        var = tk.BooleanVar(value=True) # Initially all selected
        name = f"Larva_{original_index+1}"
        larva_names_in_ui.append(name)
        check_vars.append(var)

    ax.set_title(f"Larva Trajectories (First {num_frames} Frames)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_aspect('equal')
    fig.tight_layout()

    canvas_frame = ttk.Frame(window, relief="groove", padding="5")
    canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar_frame = ttk.Frame(canvas_frame)
    toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame) # FIX: Assign toolbar
    toolbar.update()

    right_panel = ttk.Frame(window, padding="10")
    right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

    ttk.Label(right_panel, text="Uncheck larvae to exclude from export:", font=("Arial", 10, "bold")).pack(anchor='w', pady=(0, 10))

    canvas_check = tk.Canvas(right_panel, borderwidth=0, highlightthickness=0)
    canvas_check.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=canvas_check.yview)
    scrollbar.pack(side=tk.RIGHT, fill="y")

    canvas_check.configure(yscrollcommand=scrollbar.set)
    canvas_check.bind('<Configure>', lambda e: canvas_check.configure(scrollregion = canvas_check.bbox("all")))

    checkbox_frame = ttk.Frame(canvas_check)
    canvas_check.create_window((0, 0), window=checkbox_frame, anchor="nw")

    def update_plot_visibility(idx):
        if check_vars[idx].get():
            plotted_lines[idx].set_color(colors[idx % len(colors)])
            plotted_lines[idx].set_linestyle('-')
            plotted_lines[idx].set_visible(True) # Ensure it's visible
        else:
            plotted_lines[idx].set_color('lightgrey')
            plotted_lines[idx].set_linestyle('--')
            plotted_lines[idx].set_visible(True) # Keep visible even if grayed out
            
        canvas.draw_idle()

    for i, name in enumerate(larva_names_in_ui):
        chk = ttk.Checkbutton(checkbox_frame, text=name, variable=check_vars[i], command=lambda idx=i: update_plot_visibility(idx))
        chk.pack(anchor='w', pady=1)

    def export_selected_data():
        selected_pairs = []
        for i, var in enumerate(check_vars):
            if var.get():
                selected_pairs.append((larva_names_in_ui[i], detected_tracks[i][1]))

        if not selected_pairs:
            messagebox.showwarning("No Selection", "No larvae selected for export. Please check at least one larva.")
            return

        selected_names_for_export = [name for name, _ in selected_pairs]
        confirm = messagebox.askyesno(
            "Confirm Export",
            f"You are about to export data for {len(selected_names_for_export)} larvae.\n\n" + 
            ", ".join(selected_names_for_export) + "\n\nProceed with export?"
        )
        if not confirm:
            return

        selected_tracks_data_for_export = [track_data for _, track_data in selected_pairs]
        
        save_tracks_and_velocities(
            selected_tracks_data_for_export, selected_names_for_export, num_frames, tif_path,
            fps, mm_per_px, fig
        )
        messagebox.showinfo("Export Complete", "Larva paths, velocities, and plot have been saved successfully.")
        
        # window.destroy()
        sys.exit() 

    def select_all_larvae():
        for i, var in enumerate(check_vars):
            if not var.get():
                var.set(True)
                update_plot_visibility(i)

    def deselect_all_larvae():
        for i, var in enumerate(check_vars):
            if var.get():
                var.set(False)
                update_plot_visibility(i)

    def on_closing(win):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit? No data will be exported if you quit now."):
            win.destroy()
            if main_tk_root:
                main_tk_root.quit()
                main_tk_root.destroy()

    window.protocol("WM_DELETE_WINDOW", lambda: on_closing(window))

    button_frame = ttk.Frame(right_panel)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

    ttk.Button(button_frame, text="Select All", command=select_all_larvae).pack(side=tk.LEFT, expand=True, padx=2)
    ttk.Button(button_frame, text="Deselect All", command=deselect_all_larvae).pack(side=tk.LEFT, expand=True, padx=2)
    ttk.Button(button_frame, text="Export Selected", command=export_selected_data).pack(side=tk.BOTTOM, fill=tk.X, pady=(10,0))

if __name__ == '__main__':
    browse_file()
    sys.exit()