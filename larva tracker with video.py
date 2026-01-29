import os
import sys
import cv2
import numpy as np
import tifffile
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.spatial import distance_matrix
import pandas as pd

# =========================
# Globals
# =========================
main_tk_root = None

def on_root_close():
    if messagebox.askokcancel("Quit", "Quit the application?"):
        try:
            if main_tk_root:
                main_tk_root.quit()
                main_tk_root.destroy()
        finally:
            sys.exit(0)

# =========================
# Utilities
# =========================
def open_video_ffmpeg(path):
    """Open video using FFmpeg backend to avoid GStreamer plugin issues."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise IOError("Failed to open video. Try converting to MP4/H.264 or install FFmpeg-enabled OpenCV.")
    return cap

def enforce_square_roi(x, y, w, h, frame_w, frame_h):
    """Make ROI square by taking the larger side; clamp to frame."""
    size = max(w, h)
    x2 = min(x + size, frame_w)
    y2 = min(y + size, frame_h)
    # Recompute to keep size consistent
    size = min(x2 - x, y2 - y)
    return (int(x), int(y), int(size), int(size))

def apply_contrast_brightness(gray, alpha, beta):
    """
    gray: uint8 0..255, alpha: contrast scale (float), beta: brightness shift (int)
    """
    # convertScaleAbs does gray*alpha + beta and clips to 0..255
    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

def resize_to_1080_square(img):
    return cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_AREA)

def show_matplot_image_in_frame(parent, image, title=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    ax.axis('off')
    if title:
        ax.set_title(title)
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, parent)
    toolbar.update()
    return fig, ax, canvas, widget

# =========================
# Stage 1: Preprocess (Video -> Grayscale TIFF)
# =========================
class PreprocessWindow(tk.Toplevel):
    """
    Flow:
    - Select video
    - Select square ROI (OpenCV window)
    - Adjust brightness/contrast with sliders
    - Preview (first frame)
    - Process full video: crop -> resize (1080x1080) -> grayscale -> brightness/contrast -> save TIFF
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("1) Preprocess Video → Grayscale TIFF")
        self.geometry("1000x800")
        self.parent = parent

        # State
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.roi = None  # (x,y,w,h) enforced square
        self.alpha = 1.0  # contrast
        self.beta = 0     # brightness
        self.preview_frame = None
        self.processed_tiff_path = None

        # UI
        self.build_ui()

    def build_ui(self):
        topbar = ttk.Frame(self)
        topbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Button(topbar, text="Select Video", command=self.select_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(topbar, text="Select Square ROI", command=self.select_roi, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        ttk.Label(topbar, text="Contrast").pack(side=tk.LEFT, padx=(20,5))
        self.s_contrast = ttk.Scale(topbar, from_=0.5, to=3.0, value=1.0, command=self.on_contrast_change)
        self.s_contrast.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Label(topbar, text="Brightness").pack(side=tk.LEFT, padx=(20,5))
        self.s_brightness = ttk.Scale(topbar, from_=-100, to=100, value=0, command=self.on_brightness_change)
        self.s_brightness.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(topbar, text="Preview", command=self.update_preview, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)
        ttk.Button(topbar, text="Export TIFF", command=self.process_full_video, state=tk.DISABLED).pack(side=tk.LEFT, padx=5)

        self.status = ttk.Label(self, text="Select a video to begin.")
        self.status.pack(side=tk.TOP, anchor="w", padx=12)

        self.preview_frame_container = ttk.Frame(self)
        self.preview_frame_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.progress = ttk.Progressbar(self, length=600, mode='determinate')
        self.progress.pack(side=tk.BOTTOM, pady=10)

        # Next stage button
        self.btn_next = ttk.Button(self, text="Next: Threshold TIFF", command=self.go_threshold, state=tk.DISABLED)
        self.btn_next.pack(side=tk.BOTTOM, pady=(0,10))

    def select_video(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.mpg *.mpeg")]
        )
        if not path:
            return
        try:
            cap = open_video_ffmpeg(path)
        except Exception as e:
            messagebox.showerror("Error", f"{e}")
            return

        self.video_path = path
        self.cap = cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.status.config(text=f"Loaded: {os.path.basename(path)} | Frames: {self.total_frames}")

        # Enable ROI and preview
        for child in self.winfo_children():
            if isinstance(child, ttk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, ttk.Button) and btn['text'] in ("Select Square ROI", "Preview"):
                        btn['state'] = tk.NORMAL

        # Grab first frame for preview base
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.cap.read()
        if not ok:
            messagebox.showerror("Error", "Could not read first frame.")
            return
        self.preview_base = frame.copy()
        self.preview_frame = frame.copy()
        self.update_preview()

    def select_roi(self):
        if self.preview_base is None:
            return
        # Use OpenCV's ROI selector on BGR image
        temp = self.preview_base.copy()
        cv2.namedWindow("Draw ROI (ESC/Enter to close)")
        r = cv2.selectROI("Draw ROI (ESC/Enter to close)", temp, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Draw ROI (ESC/Enter to close)")
        if r == (0,0,0,0):
            return
        x, y, w, h = r
        h_img, w_img = temp.shape[:2]
        self.roi = enforce_square_roi(x, y, w, h, w_img, h_img)
        self.status.config(text=f"ROI set to (x={self.roi[0]}, y={self.roi[1]}, size={self.roi[2]})")
        self.update_preview()

        # Enable export TIFF
        for child in self.winfo_children():
            if isinstance(child, ttk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, ttk.Button) and btn['text'] == "Export TIFF":
                        btn['state'] = tk.NORMAL

    def on_contrast_change(self, val):
        self.alpha = float(val)
        self.update_preview(lazy=True)

    def on_brightness_change(self, val):
        self.beta = int(float(val))
        self.update_preview(lazy=True)

    def update_preview(self, lazy=False):
        if self.preview_base is None:
            return
        # Build pipeline on the first frame
        frame = self.preview_base.copy()
        if self.roi:
            x, y, s, _ = self.roi
            frame = frame[y:y+s, x:x+s]
        frame = resize_to_1080_square(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        proc = apply_contrast_brightness(gray, self.alpha, self.beta)

        # Clear previous plot
        for child in self.preview_frame_container.winfo_children():
            child.destroy()
        show_matplot_image_in_frame(self.preview_frame_container, proc, title="Preview (first frame)")
        if not lazy:
            self.status.config(text="Preview updated.")

    def process_full_video(self):
        if not self.cap or not self.video_path or not self.roi:
            messagebox.showwarning("Missing", "Make sure video and ROI are set.")
            return

        out_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save processed TIFF",
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif *.tiff")]
        )
        if not out_path:
            return

        x, y, s, _ = self.roi
        frames_out = []
        self.progress['maximum'] = self.total_frames
        self.progress['value'] = 0
        self.status.config(text="Processing video → TIFF (this may take a while)...")

        # Iterate frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in range(self.total_frames):
            ok, frame = self.cap.read()
            if not ok:
                break
            crop = frame[y:y+s, x:x+s]
            crop = resize_to_1080_square(crop)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            proc = apply_contrast_brightness(gray, self.alpha, self.beta)
            frames_out.append(proc.astype(np.uint8))

            if i % 10 == 0 or i == self.total_frames-1:
                self.progress['value'] = i+1
                self.update_idletasks()

        # Save TIFF
        try:
            arr = np.stack(frames_out, axis=0)
            tifffile.imwrite(out_path, arr, photometric='minisblack', imagej=True)
            self.processed_tiff_path = out_path
            messagebox.showinfo("Saved", f"Processed TIFF saved:\n{out_path}")
            self.status.config(text=f"Processed TIFF: {os.path.basename(out_path)}")

            # Enable Next
            self.btn_next['state'] = tk.NORMAL
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write TIFF: {e}")

    def go_threshold(self):
        if not self.processed_tiff_path:
            messagebox.showwarning("Missing", "Export the processed TIFF first.")
            return
        ThresholdWindow(self.parent, self.processed_tiff_path)
        self.destroy()

# =========================
# Stage 2: Threshold TIFF -> Thresholded TIFF
# =========================
class ThresholdWindow(tk.Toplevel):
    """
    Load processed grayscale TIFF and pick a threshold with slider. Save new binary TIFF.
    Then proceed to tracking.
    """
    def __init__(self, parent, tiff_path):
        super().__init__(parent)
        self.title("2) Threshold Grayscale TIFF → Binary TIFF")
        self.geometry("1000x800")
        self.parent = parent
        self.tiff_path = tiff_path
        self.frames = None
        self.th_val = 127
        self.binary_tiff_path = None
        self._load_tiff()
        self.build_ui()

    def _load_tiff(self):
        try:
            self.frames = tifffile.imread(self.tiff_path)  # [N, H, W], uint8
            if self.frames.ndim != 3:
                raise ValueError("Expected 3D grayscale TIFF (frames, H, W).")
        except Exception as e:
            messagebox.showerror("Error", f"Could not read TIFF: {e}")
            self.destroy()

    def build_ui(self):
        topbar = ttk.Frame(self)
        topbar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Label(topbar, text=f"Loaded TIFF: {os.path.basename(self.tiff_path)} | Frames: {len(self.frames)}").pack(side=tk.LEFT)
        ttk.Label(topbar, text="  Threshold: ").pack(side=tk.LEFT, padx=(20,5))
        self.s_thresh = ttk.Scale(topbar, from_=0, to=255, value=127, command=self.on_thresh_change)
        self.s_thresh.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(topbar, text="Preview", command=self.update_preview).pack(side=tk.LEFT, padx=8)
        ttk.Button(topbar, text="Export Binary TIFF", command=self.export_binary).pack(side=tk.LEFT, padx=8)

        self.status = ttk.Label(self, text="Move the threshold slider and click Preview.")
        self.status.pack(side=tk.TOP, anchor="w", padx=12)

        self.preview_frame_container = ttk.Frame(self)
        self.preview_frame_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.btn_next = ttk.Button(self, text="Next: Track Larvae", command=self.go_track, state=tk.DISABLED)
        self.btn_next.pack(side=tk.BOTTOM, pady=10)

        self.progress = ttk.Progressbar(self, length=600, mode='determinate')
        self.progress.pack(side=tk.BOTTOM, pady=(0,10))

        self.update_preview()

    def on_thresh_change(self, val):
        self.th_val = int(float(val))

    def update_preview(self):
        # Show a mid frame preview
        idx = len(self.frames)//2
        frame = self.frames[idx]
        _, binary = cv2.threshold(frame, self.th_val, 255, cv2.THRESH_BINARY)
        # Clear old
        for c in self.preview_frame_container.winfo_children():
            c.destroy()
        show_matplot_image_in_frame(self.preview_frame_container, binary, title=f"Preview frame #{idx} @ threshold={self.th_val}")
        self.status.config(text=f"Preview updated @ threshold={self.th_val}")

    def export_binary(self):
        out_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save binary TIFF",
            defaultextension=".tif",
            filetypes=[("TIFF", "*.tif *.tiff")]
        )
        if not out_path:
            return

        self.progress['maximum'] = len(self.frames)
        self.progress['value'] = 0
        binaries = []
        for i, frame in enumerate(self.frames):
            _, binary = cv2.threshold(frame, self.th_val, 255, cv2.THRESH_BINARY)
            binaries.append(binary.astype(np.uint8))
            if i % 25 == 0 or i == len(self.frames)-1:
                self.progress['value'] = i+1
                self.update_idletasks()

        try:
            arr = np.stack(binaries, axis=0)
            tifffile.imwrite(out_path, arr, photometric='minisblack', imagej=True)
            self.binary_tiff_path = out_path
            messagebox.showinfo("Saved", f"Binary TIFF saved:\n{out_path}")
            self.btn_next['state'] = tk.NORMAL
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write binary TIFF: {e}")

    def go_track(self):
        if not self.binary_tiff_path:
            messagebox.showwarning("Missing", "Export the binary TIFF first.")
            return
        TrackingWindow(self.parent, self.binary_tiff_path)
        self.destroy()

# =========================
# Stage 3: Tracking (as before)
# =========================
class TrackingWindow(tk.Toplevel):
    """
    Tracks larvae across frames of a (binary) TIFF, computes velocities, and exports CSVs/plots.
    """
    def __init__(self, parent, tiff_path):
        super().__init__(parent)
        self.title("3) Track Larvae")
        self.geometry("1200x900")
        self.parent = parent
        self.tiff_path = tiff_path

        self.frames = None
        self.num_frames = 0
        self.fps = None
        self.mm_per_px = None

        self.detected_tracks = None  # list of (idx, [(x,y),...])
        self.colors = plt.cm.tab10.colors

        self._load_frames()
        self._ask_parameters()
        if self.num_frames:
            self._run_tracking()
            self._filter_and_plot()
        else:
            self.destroy()

    def _load_frames(self):
        try:
            frames = tifffile.imread(self.tiff_path)
            if frames.ndim != 3:
                raise ValueError("Expected a 3D TIFF (frames, H, W).")
            # Ensure uint8
            if frames.dtype != np.uint8:
                frames = cv2.normalize(frames, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            self.frames = frames
        except Exception as e:
            messagebox.showerror("Error", f"Could not read TIFF: {e}")
            self.destroy()

    def _ask_parameters(self):
        if self.frames is None:
            return
        total_frames = len(self.frames)
        self.num_frames = simpledialog.askinteger(
            "Frames to Analyze",
            f"Total frames in TIFF: {total_frames}\nEnter number of frames to analyze:",
            minvalue=1, maxvalue=total_frames, parent=self
        )
        if not self.num_frames:
            messagebox.showinfo("Cancelled", "No frame count provided.")
            return
        self.fps = simpledialog.askfloat(
            "Frame Rate (FPS)",
            "Enter frame rate (fps) used during acquisition:",
            minvalue=0.1, parent=self
        )
        self.mm_per_px = simpledialog.askfloat(
            "Conversion Factor",
            "Enter conversion factor (mm per pixel):",
            minvalue=0.0001, parent=self
        )

    def _run_tracking(self):
        max_match_distance = 50
        frames = self.frames[:self.num_frames]

        # Initial detections from frame 0 (binary)
        frame0 = frames[0]
        _, thresh0 = cv2.threshold(frame0, 0, 255, cv2.THRESH_BINARY)  # ensure binary
        contours0, _ = cv2.findContours(thresh0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours0:
            messagebox.showwarning("No Larvae", "No larvae detected in first frame.")
            self.destroy()
            return

        larva_tracks = []
        for cnt in contours0:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                larva_tracks.append([(cx, cy)])

        # Progress UI
        self.progress = ttk.Progressbar(self, length=600, mode='determinate', maximum=self.num_frames-1)
        self.progress.pack(side=tk.TOP, pady=10)
        self.update_idletasks()

        for i in range(1, self.num_frames):
            frame = frames[i]
            _, thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            current_centroids = []
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    current_centroids.append((cx, cy))

            previous_positions = [track[-1] for track in larva_tracks]
            if current_centroids and previous_positions:
                dist_mat = distance_matrix(previous_positions, current_centroids)
                matches = [-1] * len(previous_positions)
                current_matched = [False] * len(current_centroids)
                flat_indices = dist_mat.argsort(axis=None)
                for flat_idx in flat_indices:
                    prev_idx = flat_idx // dist_mat.shape[1]
                    curr_idx = flat_idx % dist_mat.shape[1]
                    dist_val = dist_mat[prev_idx, curr_idx]
                    if dist_val < max_match_distance and not current_matched[curr_idx] and matches[prev_idx] == -1:
                        larva_tracks[prev_idx].append(current_centroids[curr_idx])
                        matches[prev_idx] = curr_idx
                        current_matched[curr_idx] = True
                for prev_idx, matched in enumerate(matches):
                    if matched == -1:
                        larva_tracks[prev_idx].append((np.nan, np.nan))
            else:
                for track in larva_tracks:
                    track.append((np.nan, np.nan))

            self.progress['value'] = i
            self.update_idletasks()

        # Pad
        for track in larva_tracks:
            while len(track) < self.num_frames:
                track.append((np.nan, np.nan))

        # Store with original indices
        self.detected_tracks = [(i, tr) for i, tr in enumerate(larva_tracks)]

    def _filter_and_plot(self):
        if not self.detected_tracks:
            return
        # Filter tracks with no NaNs
        consistent = []
        removed = 0
        for i, tr in self.detected_tracks:
            arr = np.array(tr)
            if not np.any(np.isnan(arr)):
                consistent.append((i, tr))
            else:
                removed += 1

        if not consistent:
            messagebox.showwarning("No Consistent Larvae", "No larvae were consistently detected across all frames.")
            self.destroy()
            return

        messagebox.showinfo("Filtering", f"Filtered out {removed} tracks with gaps. {len(consistent)} remain.")

        # Build UI for selection and export
        self._launch_plot_and_selection_ui(consistent)

    def _launch_plot_and_selection_ui(self, tracks_filtered):
        # Plot
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        fig, ax = plt.subplots(figsize=(7, 7))
        lines = []
        names = []
        check_vars = []

        for i, (idx, tr) in enumerate(tracks_filtered):
            arr = np.array(tr)
            color = self.colors[i % len(self.colors)]
            line, = ax.plot(arr[:, 0], arr[:, 1], '-', marker='o', markersize=2, label=f"Larva_{idx+1}", color=color)
            ax.text(arr[-1, 0], arr[-1, 1], f"L{idx+1}", fontsize=8, color=color, ha='left', va='center')
            lines.append(line)
            names.append(f"Larva_{idx+1}")
            check_vars.append(tk.BooleanVar(value=True))

        ax.set_title(f"Larva Trajectories (First {self.num_frames} Frames)")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.invert_yaxis()
        ax.grid(True)
        ax.set_aspect('equal')
        fig.tight_layout()

        canvas_frame = ttk.Frame(frame, relief="groove", padding=5)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar_frame = ttk.Frame(canvas_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        NavigationToolbar2Tk(canvas, toolbar_frame).update()

        right_panel = ttk.Frame(frame, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(right_panel, text="Uncheck larvae to exclude:", font=("Arial", 10, "bold")).pack(anchor='w', pady=(0, 10))

        list_canvas = tk.Canvas(right_panel, borderwidth=0, highlightthickness=0)
        list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=list_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        list_canvas.configure(yscrollcommand=scrollbar.set)

        inner = ttk.Frame(list_canvas)
        list_canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: list_canvas.configure(scrollregion=list_canvas.bbox("all")))

        def update_plot(i):
            if check_vars[i].get():
                lines[i].set_color(self.colors[i % len(self.colors)])
                lines[i].set_linestyle('-')
                lines[i].set_visible(True)
            else:
                lines[i].set_color('lightgrey')
                lines[i].set_linestyle('--')
                lines[i].set_visible(True)
            canvas.draw_idle()

        for i, nm in enumerate(names):
            chk = ttk.Checkbutton(inner, text=nm, variable=check_vars[i], command=lambda i=i: update_plot(i))
            chk.pack(anchor='w', pady=1)

        # Buttons
        btns = ttk.Frame(right_panel)
        btns.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        def select_all():
            for i, v in enumerate(check_vars):
                if not v.get():
                    v.set(True)
                    update_plot(i)

        def deselect_all():
            for i, v in enumerate(check_vars):
                if v.get():
                    v.set(False)
                    update_plot(i)

        def export_selected():
            selected_pairs = []
            for i, v in enumerate(check_vars):
                if v.get():
                    selected_pairs.append((names[i], tracks_filtered[i][1]))
            if not selected_pairs:
                messagebox.showwarning("No Selection", "No larvae selected.")
                return
            sel_names = [n for n, _ in selected_pairs]
            sel_tracks = [t for _, t in selected_pairs]
            self._save_tracks_and_velocities(sel_tracks, sel_names, fig)
            messagebox.showinfo("Export", "Paths, velocities, and plots saved.")
            self.destroy()

        ttk.Button(btns, text="Select All", command=select_all).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(btns, text="Deselect All", command=deselect_all).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(right_panel, text="Export Selected", command=export_selected).pack(side=tk.BOTTOM, fill=tk.X, pady=(10,0))

    # ---- Exports ----
    def _calculate_velocities(self, tracks, names):
        out = []
        total_time = (self.num_frames / self.fps) if (self.fps is not None and self.fps > 0) else np.nan
        for nm, tr in zip(names, tracks):
            arr = np.array(tr)
            diffs = np.diff(arr, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            path_px = np.nansum(dists)
            v_px_s = path_px / total_time if total_time and total_time > 0 else np.nan
            v_mm_s = v_px_s * self.mm_per_px if (self.mm_per_px is not None and not np.isnan(v_px_s)) else np.nan
            out.append({
                "Larva": nm,
                "Path Length (px)": round(path_px, 2),
                "Velocity (px/s)": round(v_px_s, 2) if not np.isnan(v_px_s) else "N/A",
                "Velocity (mm/s)": round(v_mm_s, 2) if (not isinstance(v_mm_s, float) or not np.isnan(v_mm_s)) else "N/A"
            })
        return out

    def _save_tracks_and_velocities(self, tracks, names, fig):
        base = os.path.splitext(os.path.basename(self.tiff_path))[0]
        out_dir = os.path.dirname(self.tiff_path)

        # Paths CSV
        data = {"Frame": list(range(self.num_frames))}
        for tr, nm in zip(tracks, names):
            xs = [p[0] for p in tr]
            ys = [p[1] for p in tr]
            data[f"{nm}_X"] = xs
            data[f"{nm}_Y"] = ys
        df_paths = pd.DataFrame(data)
        paths_csv = os.path.join(out_dir, f"{base}_larva_paths.csv")
        df_paths.to_csv(paths_csv, index=False)

        # Velocities CSV
        if self.fps is not None and self.fps > 0:
            vels = self._calculate_velocities(tracks, names)
            df_vels = pd.DataFrame(vels)
            vels_csv = os.path.join(out_dir, f"{base}_velocities.csv")
            df_vels.to_csv(vels_csv, index=False)

        # Plot with axes
        plot_png = os.path.join(out_dir, f"{base}_trajectories.png")
        fig.savefig(plot_png, dpi=300)

        # Clean overlay (transparent) sized to frame bounds
        try:
            allx, ally = [], []
            for tr in tracks:
                arr = np.array(tr)
                allx.extend(arr[:,0][~np.isnan(arr[:,0])])
                ally.extend(arr[:,1][~np.isnan(arr[:,1])])
            if len(allx) > 0 and len(ally) > 0:
                w = int(np.nanmax(allx)) + 5
                h = int(np.nanmax(ally)) + 5
                dpi = 100
                f2, ax2 = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
                ax2.set_xlim(0, w)
                ax2.set_ylim(h, 0)
                ax2.set_xlabel("X (pixels)")
                ax2.set_ylabel("Y (pixels)")
                for i, tr in enumerate(tracks):
                    arr = np.array(tr)
                    ax2.plot(arr[:,0], arr[:,1], '-', linewidth=1.5, color=self.colors[i % len(self.colors)])
                overlay_png = os.path.join(out_dir, f"{base}_trajectories_overlay.png")
                f2.savefig(overlay_png, dpi=dpi, transparent=True)
                plt.close(f2)
        except Exception as e:
            print(f"[Warning] Could not create overlay: {e}")

# =========================
# App Launcher (Stages 1→2→3)
# =========================
class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        global main_tk_root
        main_tk_root = self
        self.title("Larva Tracking Pipeline")
        self.geometry("500x250")
        self.protocol("WM_DELETE_WINDOW", on_root_close)

        ttk.Label(self, text="Flow:", font=("Arial", 11, "bold")).pack(pady=(20,5))
        ttk.Label(self, text="1) Select Video → Crop ROI (square) → 1080×1080 → Grayscale → Adjust B/C → Export TIFF").pack()
        ttk.Label(self, text="2) Load TIFF → Adjust Threshold → Export Binary TIFF").pack()
        ttk.Label(self, text="3) Track Larvae on Binary TIFF → Export CSVs + Plots").pack(pady=(0,20))

        ttk.Button(self, text="Begin", command=self.start).pack()

    def start(self):
        PreprocessWindow(self)

# =========================
# Main
# =========================
if __name__ == "__main__":
    app = Launcher()
    app.mainloop()
