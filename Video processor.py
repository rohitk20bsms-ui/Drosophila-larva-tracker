
import os
import sys
import math
import threading
import cv2
import numpy as np
import tifffile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------
# Utility helpers
# -------------------------
def open_video_capture(path: str):
    """Try FFmpeg backend first, fallback to default if needed."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not (cap and cap.isOpened()):
        cap2 = cv2.VideoCapture(path)
        if not (cap2 and cap2.isOpened()):
            raise IOError("Cannot open video. Try converting to MP4 or ensure OpenCV is built with FFmpeg.")
        return cap2
    return cap

def gray_if_needed(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()

def resize_1080(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((1080,1080), dtype=np.uint8)
    return cv2.resize(img, (1080,1080), interpolation=cv2.INTER_AREA)

def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros((1080,1080), dtype=np.uint8)
    if img.dtype == np.uint8:
        return img
    arr = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return arr.astype(np.uint8)

def safe_destroy_opencv():
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

# -------------------------
# Square ROI Editor (OpenCV)
# -------------------------
class SquareROIEditor:
    """
    Interactive editor that allows dragging 4 corners.
    Press ENTER to accept, ESC to cancel. 'r' to reset to centered square.
    Returns 4x2 numpy array of corner coordinates (clockwise or similar).
    """
    def __init__(self, frame_bgr: np.ndarray):
        self.frame = frame_bgr.copy()
        h, w = self.frame.shape[:2]
        side = min(h, w) // 3
        cx, cy = w // 2, h // 2
        x1 = cx - side // 2
        y1 = cy - side // 2
        x2 = x1 + side
        y2 = y1 + side
        self.corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        self.drag_idx = None
        self.radius = max(6, int(min(h, w) * 0.015))
        self.win_name = "Square ROI Editor - drag corners | Enter accept | Esc cancel | r reset"
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self._mouse_cb)

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            dists = [ (x-cx)**2 + (y-cy)**2 for cx,cy in self.corners ]
            idx = int(np.argmin(dists))
            if dists[idx] <= (self.radius * self.radius * 9):
                self.drag_idx = idx
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx is not None:
            # clamp coordinates
            self.corners[self.drag_idx][0] = int(max(0, min(x, self.frame.shape[1]-1)))
            self.corners[self.drag_idx][1] = int(max(0, min(y, self.frame.shape[0]-1)))

    def _draw(self):
        disp = self.frame.copy()
        pts = np.array(self.corners, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(disp, [pts], isClosed=True, color=(0,255,0), thickness=2)
        for i,(cx,cy) in enumerate(self.corners):
            cv2.circle(disp, (int(cx),int(cy)), self.radius, (0,0,255), -1)
            cv2.putText(disp, str(i+1), (int(cx)+6, int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return disp

    def run(self):
        while True:
            disp = self._draw()
            cv2.imshow(self.win_name, disp)
            key = cv2.waitKey(20) & 0xFF
            if key == 13:  # Enter
                cv2.destroyWindow(self.win_name)
                return np.array(self.corners, dtype=np.int32)
            elif key == 27:  # ESC
                cv2.destroyWindow(self.win_name)
                return None
            elif key == ord('r'):
                # reset to centered square
                h,w = self.frame.shape[:2]
                side = min(h,w)//3
                cx,cy = w//2, h//2
                x1 = cx-side//2; y1 = cy-side//2; x2 = x1+side; y2 = y1+side
                self.corners = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

# -------------------------
# Trim Dialog with live previews
# -------------------------
class TrimDialog:
    """
    Modal dialog to choose start/end in seconds via sliders.
    Shows live thumbnails for the first and last frames of the chosen range.
    preview_frame_callback(frame_index) -> main app will provide a frame (BGR numpy) to display.
    """
    def __init__(self, parent, total_frames: int, fps: float, start_sec_default: float, end_sec_default: float, preview_callback):
        self.parent = parent
        self.total_frames = int(total_frames)
        self.fps = float(fps) if fps and fps>0 else 1.0
        self.duration = self.total_frames / self.fps
        self.preview_callback = preview_callback  # function receiving frame_index -> returns BGR frame
        self.result = None

        self.top = tk.Toplevel(parent)
        self.top.title("Trim Video (seconds) - live preview")
        self.top.geometry("1000x600")
        self.top.transient(parent)
        self.top.grab_set()

        # Variables: seconds as float stored in Tk DoubleVar
        self.start_sec = tk.DoubleVar(value=max(0.0, float(start_sec_default)))
        self.end_sec = tk.DoubleVar(value=min(self.duration, float(end_sec_default)))

        self._build_ui()
        # initial preview
        self._update_previews()

    def _build_ui(self):
        info = ttk.Frame(self.top)
        info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(info, text=f"Total frames: {self.total_frames}").pack(side=tk.LEFT, padx=6)
        ttk.Label(info, text=f"FPS: {self.fps:.3f}").pack(side=tk.LEFT, padx=6)
        ttk.Label(info, text=f"Duration: {self.duration:.2f} s").pack(side=tk.LEFT, padx=6)

        # preview frames container (left = start, right = end)
        pv = ttk.Frame(self.top)
        pv.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = ttk.LabelFrame(pv, text="Start Frame Preview")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=4)
        right = ttk.LabelFrame(pv, text="End Frame Preview")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.left_canvas = ttk.Frame(left)
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        self.right_canvas = ttk.Frame(right)
        self.right_canvas.pack(fill=tk.BOTH, expand=True)

        # sliders area
        sliders = ttk.Frame(self.top)
        sliders.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(sliders, text="Start (s)").grid(row=0, column=0, sticky='w', padx=4)
        self.start_scale = ttk.Scale(sliders, from_=0.0, to=self.duration, variable=self.start_sec, orient='horizontal', length=600, command=self._on_start_change)
        self.start_scale.grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(sliders, text="End (s)").grid(row=1, column=0, sticky='w', padx=4)
        self.end_scale = ttk.Scale(sliders, from_=0.0, to=self.duration, variable=self.end_sec, orient='horizontal', length=600, command=self._on_end_change)
        self.end_scale.grid(row=1, column=1, padx=4, pady=4)

        # labels for numeric display
        bottom_info = ttk.Frame(self.top)
        bottom_info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.range_label = ttk.Label(bottom_info, text="")
        self.range_label.pack(side=tk.LEFT, padx=6)
        ttk.Button(bottom_info, text="OK", command=self._on_ok).pack(side=tk.RIGHT, padx=6)
        ttk.Button(bottom_info, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=6)

        self._update_range_label()

    def _on_start_change(self, val):
        s = float(self.start_sec.get())
        e = float(self.end_sec.get())
        if s > e:
            # clamp end to start
            self.end_sec.set(s)
        self._update_range_label()
        self._update_previews()

    def _on_end_change(self, val):
        s = float(self.start_sec.get())
        e = float(self.end_sec.get())
        if e < s:
            self.start_sec.set(e)
        self._update_range_label()
        self._update_previews()

    def _update_range_label(self):
        s = float(self.start_sec.get())
        e = float(self.end_sec.get())
        start_frame = min(self.total_frames-1, max(0, int(round(s * self.fps))))
        end_frame = min(self.total_frames-1, max(0, int(round(e * self.fps))))
        length_frames = max(0, end_frame - start_frame + 1)
        length_sec = (length_frames / self.fps) if self.fps>0 else 0.0
        self.range_label.config(text=f"Selected: {s:.3f}s → {e:.3f}s  |  Frames: {start_frame} → {end_frame}  ({length_frames} frames, {length_sec:.3f} s)")

    def _update_previews(self):
        # ask main app to provide frame for given index
        s = float(self.start_sec.get()); e = float(self.end_sec.get())
        start_idx = min(self.total_frames-1, max(0, int(round(s * self.fps))))
        end_idx = min(self.total_frames-1, max(0, int(round(e * self.fps))))
        # call preview callback to get frames; callback should return BGR frame or None
        try:
            frame_s = self.preview_callback(start_idx)
        except Exception:
            frame_s = None
        try:
            frame_e = self.preview_callback(end_idx)
        except Exception:
            frame_e = None
        # display in respective canvases
        self._display_in_frame(self.left_canvas, frame_s, title=f"Start frame #{start_idx}")
        self._display_in_frame(self.right_canvas, frame_e, title=f"End frame #{end_idx}")
        self._update_range_label()

    def _display_in_frame(self, container: ttk.Frame, frame_bgr, title: str="Preview"):
        for w in container.winfo_children():
            w.destroy()
        if frame_bgr is None:
            lbl = ttk.Label(container, text="(frame unavailable)")
            lbl.pack(expand=True)
            return
        # create grayscale preview resized to ~420x420 while preserving content center
        frame = frame_bgr.copy()
        gray = gray_if_needed(frame)
        h,w = gray.shape[:2]
        side = min(h,w)
        cx,cy = w//2, h//2
        x1 = max(0, cx-side//2); y1 = max(0, cy-side//2)
        crop = gray[y1:y1+side, x1:x1+side]
        try:
            thumb = cv2.resize(crop, (420,420), interpolation=cv2.INTER_AREA)
        except Exception:
            thumb = cv2.resize(crop, (420,420))
        # convert to RGB for matplotlib
        rgb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
        fig = plt.Figure(figsize=(3.5,3.5))
        ax = fig.add_subplot(111)
        ax.imshow(rgb, cmap='gray')
        ax.axis('off')
        ax.set_title(title)
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_ok(self):
        s = float(self.start_sec.get()); e = float(self.end_sec.get())
        s_idx = min(self.total_frames-1, max(0, int(round(s * self.fps))))
        e_idx = min(self.total_frames-1, max(0, int(round(e * self.fps))))
        if e_idx < s_idx:
            messagebox.showwarning("Invalid trim", "End time is before start time.")
            return
        self.result = (s, e)
        try:
            self.top.destroy()
        except Exception:
            pass

    def _on_cancel(self):
        self.result = None
        try:
            self.top.destroy()
        except Exception:
            pass

# -------------------------
# Main App
# -------------------------
class VideoPreprocessorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Preprocessor - Square ROI, Trim (seconds), Threshold, Export TIFF")
        self.root.geometry("1200x820")
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.first_frame = None

        # ROI state
        self.square_pts = None  # 4x2 array of corners in source coordinates
        self.square_bbox = None  # x1,y1,x2,y2

        # trim state
        self.start_sec = 0.0
        self.end_sec = 0.0

        # processing params
        self.alpha = tk.DoubleVar(value=1.0)
        self.beta = tk.IntVar(value=0)
        self.threshold = tk.IntVar(value=127)
        self.invert_lut = tk.BooleanVar(value=False)

        # export control
        self._export_thread = None
        self._export_cancel = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(top, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=6)
        self.open_label = ttk.Label(top, text="No video loaded")
        self.open_label.pack(side=tk.LEFT, padx=12)

        # ROI edit
        self.roi_btn = ttk.Button(top, text="Edit Square ROI", command=self.edit_roi, state='disabled')
        self.roi_btn.pack(side=tk.LEFT, padx=6)

        # Trim button
        self.trim_btn = ttk.Button(top, text="Trim (seconds)...", command=self.open_trim_dialog, state='disabled')
        self.trim_btn.pack(side=tk.LEFT, padx=6)

        # processing controls
        params = ttk.Frame(self.root)
        params.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(params, text="Contrast").grid(row=0, column=0, padx=6, sticky='w')
        ttk.Scale(params, from_=0.2, to=3.0, variable=self.alpha, orient='horizontal', length=220, command=lambda v: self.preview()).grid(row=0, column=1, padx=6)
        ttk.Label(params, text="Brightness").grid(row=0, column=2, padx=6, sticky='w')
        ttk.Scale(params, from_=-100, to=100, variable=self.beta, orient='horizontal', length=220, command=lambda v: self.preview()).grid(row=0, column=3, padx=6)
        ttk.Label(params, text="Threshold").grid(row=0, column=4, padx=6, sticky='w')
        ttk.Scale(params, from_=0, to=255, variable=self.threshold, orient='horizontal', length=220, command=lambda v: self.preview()).grid(row=0, column=5, padx=6)
        ttk.Checkbutton(params, text="Invert LUT", variable=self.invert_lut, command=self.preview).grid(row=0, column=6, padx=12)

        actions = ttk.Frame(self.root)
        actions.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.preview_btn = ttk.Button(actions, text="Preview first frame", command=self.preview, state='disabled')
        self.preview_btn.pack(side=tk.LEFT, padx=6)
        self.export_btn = ttk.Button(actions, text="Export TIFF (trimmed)", command=self.export_tiff, state='disabled')
        self.export_btn.pack(side=tk.LEFT, padx=6)
        self.cancel_btn = ttk.Button(actions, text="Cancel Export", command=self.cancel_export, state='disabled')
        self.cancel_btn.pack(side=tk.LEFT, padx=6)

        # preview area
        self.preview_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self._clear_preview_area()

        # status bar
        self.status_var = tk.StringVar(value="Load a video to begin.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w').pack(side=tk.BOTTOM, fill=tk.X)

    def _clear_preview_area(self):
        for c in self.preview_frame.winfo_children():
            c.destroy()
        lbl = ttk.Label(self.preview_frame, text="Preview will appear here", anchor='center')
        lbl.pack(expand=True, fill=tk.BOTH)

    def open_video(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v *.mpeg")])
        if not path:
            return
        try:
            cap = open_video_capture(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open video:\n{e}")
            return
        self.video_path = path
        self.cap = cap
        try:
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps_val = cap.get(cv2.CAP_PROP_FPS)
            self.fps = float(fps_val) if fps_val and fps_val>0 else 30.0
        except Exception:
            self.total_frames = 0
            self.fps = 30.0

        # read first frame for preview and default ROI
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            messagebox.showerror("Error", "Could not read first frame from video.")
            return
        self.first_frame = frame.copy()

        # default ROI: centered square third of min dimension
        h,w = self.first_frame.shape[:2]
        side = min(h,w) // 3
        cx,cy = w//2, h//2
        x1 = cx - side//2; y1 = cy - side//2; x2 = x1 + side; y2 = y1 + side
        self.square_pts = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32)
        self.square_bbox = (x1,y1,x2,y2)

        # default trim full video
        self.start_sec = 0.0
        self.end_sec = max(0.0, self.total_frames/self.fps if self.total_frames>0 else 0.0)

        # enable UI controls
        self.open_label.config(text=f"{os.path.basename(path)} | {self.total_frames} frames | {self.fps:.2f} fps")
        self.roi_btn['state'] = 'normal'
        self.trim_btn['state'] = 'normal'
        self.preview_btn['state'] = 'normal'
        self.export_btn['state'] = 'normal'
        self.preview()

    def edit_roi(self):
        if self.first_frame is None:
            messagebox.showwarning("No video", "Open a video first.")
            return
        editor = SquareROIEditor(self.first_frame)
        pts = editor.run()
        if pts is None:
            self.status_var.set("ROI edit cancelled.")
            return
        self.square_pts = pts
        xs = pts[:,0]; ys = pts[:,1]
        x1,x2 = int(xs.min()), int(xs.max()); y1,y2 = int(ys.min()), int(ys.max())
        self.square_bbox = (x1,y1,x2,y2)
        self.status_var.set(f"Square ROI set: {self.square_bbox}")
        self.preview()

    def open_trim_dialog(self):
        if self.total_frames <= 0:
            messagebox.showwarning("No video", "Open a video first.")
            return
        td = TrimDialog(self.root, total_frames=self.total_frames, fps=self.fps,
                        start_sec_default=self.start_sec, end_sec_default=self.end_sec,
                        preview_callback=self._frame_getter_for_trim_preview)
        self.root.wait_window(td.top)
        if td.result is None:
            self.status_var.set("Trim cancelled.")
            return
        s,e = td.result
        self.start_sec = float(s); self.end_sec = float(e)
        self.status_var.set(f"Trim set: {self.start_sec:.3f}s → {self.end_sec:.3f}s")
        self.preview()

    def _frame_getter_for_trim_preview(self, frame_index: int):
        """Return BGR frame at index for trim dialog previews. Attempts to read using a short-lived VideoCapture for safety."""
        if self.video_path is None:
            return None
        try:
            cap = open_video_capture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
            ok, frame = cap.read()
            cap.release()
            if not ok:
                return None
            return frame
        except Exception:
            return None

    def _compute_crop_from_roi(self, frame_bgr):
        """Return cropped region (square) using current square_bbox if available, else center square."""
        if frame_bgr is None:
            return None
        h,w = frame_bgr.shape[:2]
        if self.square_bbox is not None:
            x1,y1,x2,y2 = self.square_bbox
            # clamp
            x1 = max(0,min(x1,w-1)); x2 = max(0,min(x2,w))
            y1 = max(0,min(y1,h-1)); y2 = max(0,min(y2,h))
            if x2<=x1 or y2<=y1:
                # fallback to center square
                side = min(h,w); cx,cy = w//2,h//2
                x1 = max(0, cx-side//2); y1 = max(0, cy-side//2); x2=x1+side; y2=y1+side
                return frame_bgr[y1:y2, x1:x2].copy()
            return frame_bgr[y1:y2, x1:x2].copy()
        else:
            side = min(h,w); cx,cy = w//2,h//2
            x1 = max(0, cx-side//2); y1 = max(0, cy-side//2)
            return frame_bgr[y1:y1+side, x1:x1+side].copy()

    def preview(self):
        """Preview first frame with current ROI, BC, threshold, invert applied (live)."""
        if self.first_frame is None:
            return
        frame = self.first_frame.copy()
        crop = self._compute_crop_from_roi(frame)
        try:
            out = resize_1080(crop)
        except Exception:
            out = cv2.resize(crop, (1080,1080), interpolation=cv2.INTER_AREA)
        gray = gray_if_needed(out)
        alpha = float(self.alpha.get()); beta = int(self.beta.get())
        proc = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        _, binary = cv2.threshold(proc, int(self.threshold.get()), 255, cv2.THRESH_BINARY)
        if self.invert_lut.get():
            binary = cv2.bitwise_not(binary)
        display_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        # draw small overlay of ROI corner indices in preview for guidance
        # Map original square pts to preview coordinates
        if self.square_bbox is not None and self.square_pts is not None:
            x1,y1,x2,y2 = self.square_bbox
            bw = x2 - x1; bh = y2 - y1
            if bw>0 and bh>0:
                sx = 1080 / bw; sy = 1080 / bh
                for i,(px,py) in enumerate(self.square_pts):
                    ox = int((px - x1)*sx)
                    oy = int((py - y1)*sy)
                    cv2.circle(display_rgb, (ox,oy), 6, (0,255,0), -1)
                    cv2.putText(display_rgb, str(i+1), (ox+6, oy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # display
        self._display_in_main_preview(display_rgb)

    def _display_in_main_preview(self, img_rgb):
        for c in self.preview_frame.winfo_children():
            c.destroy()
        fig = plt.Figure(figsize=(6.5,6.5))
        ax = fig.add_subplot(111)
        ax.imshow(img_rgb)
        ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_tiff(self):
        if self.video_path is None:
            messagebox.showwarning("No video", "Open a video first.")
            return
        out_path = filedialog.asksaveasfilename(parent=self.root, defaultextension=".tif", filetypes=[("TIFF","*.tif *.tiff")], title="Save processed TIFF (contiguous 8-bit grayscale)")
        if not out_path:
            return

        # compute frame indices
        if self.total_frames <= 0:
            messagebox.showwarning("No frames", "Video has no frames to export.")
            return
        s_idx = int(round(self.start_sec * self.fps))
        e_idx = int(round(self.end_sec * self.fps))
        s_idx = max(0, min(self.total_frames-1, s_idx))
        e_idx = max(0, min(self.total_frames-1, e_idx))
        if e_idx < s_idx:
            messagebox.showwarning("Trim error", "Invalid trim range.")
            return
        total_to_write = e_idx - s_idx + 1

        if self._export_thread and self._export_thread.is_alive():
            messagebox.showwarning("Export busy", "An export is already running.")
            return

        # launch thread
        self._export_cancel = False
        self.cancel_btn['state'] = 'normal'
        self._export_thread = threading.Thread(target=self._export_worker, args=(out_path, s_idx, e_idx, total_to_write), daemon=True)
        self._export_thread.start()

    def cancel_export(self):
        if self._export_thread and self._export_thread.is_alive():
            self._export_cancel = True
            self.status_var.set("Cancel requested...")

    def _export_worker(self, out_path, s_idx, e_idx, total_to_write):
        # small progress window
        prog = tk.Toplevel(self.root)
        prog.title("Exporting to TIFF")
        ttk.Label(prog, text=f"Exporting frames {s_idx} → {e_idx} ({total_to_write} frames)").pack(padx=8, pady=(8,4))
        pbar = ttk.Progressbar(prog, length=700, mode='determinate', maximum=total_to_write)
        pbar.pack(padx=8, pady=(0,8))
        status_label = ttk.Label(prog, text="Starting...")
        status_label.pack(padx=8, pady=(0,8))

        try:
            cap = open_video_capture(self.video_path)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open video for export:\n{e}")
            try:
                prog.destroy()
            except Exception:
                pass
            self.cancel_btn['state'] = 'disabled'
            return

        try:
            with tifffile.TiffWriter(out_path, bigtiff=True) as tw:
                # position to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
                written = 0
                idx = s_idx
                alpha = float(self.alpha.get()); beta = int(self.beta.get())
                thr = int(self.threshold.get()); invert = bool(self.invert_lut.get())
                while idx <= e_idx:
                    if self._export_cancel:
                        status_label.config(text="Cancelled by user.")
                        break
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    # crop using ROI or center
                    crop = self._compute_crop_from_roi(frame)
                    try:
                        out_img = resize_1080(crop)
                    except Exception:
                        out_img = cv2.resize(crop, (1080,1080), interpolation=cv2.INTER_AREA)
                    gray = gray_if_needed(out_img)
                    proc = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
                    _, binary = cv2.threshold(proc, thr, 255, cv2.THRESH_BINARY)
                    if invert:
                        binary = cv2.bitwise_not(binary)
                    final = ensure_uint8(binary)
                    # write contiguous page (contiguous=True) - safe for ImageJ and other readers
                    tw.write(final, contiguous=True, compression='none')
                    written += 1
                    idx += 1
                    if (written % 5 == 0) or (written == total_to_write):
                        pbar['value'] = written
                        status_label.config(text=f"Wrote {written}/{total_to_write}")
                        prog.update_idletasks()
                # end loop
            if self._export_cancel:
                messagebox.showinfo("Export", "Export cancelled.")
                self.status_var.set("Export cancelled.")
            else:
                messagebox.showinfo("Export", f"Export completed: {out_path} ({written} frames)")
                self.status_var.set(f"Exported {written} frames to {os.path.basename(out_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed exporting TIFF:\n{e}")
            self.status_var.set("Export failed.")
        finally:
            try:
                cap.release()
            except Exception:
                pass
            try:
                prog.destroy()
            except Exception:
                pass
            self.cancel_btn['state'] = 'disabled'
            self._export_cancel = False

    def _on_close(self):
        if messagebox.askokcancel("Quit", "Quit the application?"):
            try:
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                safe_destroy_opencv()
                plt.close('all')
            finally:
                try:
                    self.root.destroy()
                except Exception:
                    pass
                os._exit(0)

# -------------------------
# Run script
# -------------------------
def main():
    root = tk.Tk()
    app = VideoPreprocessorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
