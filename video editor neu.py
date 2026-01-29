import os
import sys
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
# Square ROI Editor
# -------------------------
class SquareROIEditor:
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
                h,w = self.frame.shape[:2]
                side = min(h,w)//3
                cx,cy = w//2, h//2
                x1 = cx-side//2; y1 = cy-side//2; x2 = x1+side; y2 = y1+side
                self.corners = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

# -------------------------
# Trim Dialog
# -------------------------
class TrimDialog:
    def __init__(self, parent, total_frames: int, fps: float, start_sec_default: float, end_sec_default: float, preview_callback):
        self.parent = parent
        self.total_frames = int(total_frames)
        self.fps = float(fps) if fps>0 else 1.0
        self.duration = self.total_frames / self.fps
        self.preview_callback = preview_callback
        self.result = None

        self.top = tk.Toplevel(parent)
        self.top.title("Trim Video (seconds) - live preview")
        self.top.geometry("1000x600")
        self.top.transient(parent)
        self.top.grab_set()

        self.start_sec = tk.DoubleVar(value=max(0.0, float(start_sec_default)))
        self.end_sec = tk.DoubleVar(value=min(self.duration, float(end_sec_default)))

        self._build_ui()
        self._update_previews()

    def _build_ui(self):
        # top info
        info = ttk.Frame(self.top)
        info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(info, text=f"Total frames: {self.total_frames}").pack(side=tk.LEFT, padx=6)
        ttk.Label(info, text=f"FPS: {self.fps:.3f}").pack(side=tk.LEFT, padx=6)
        ttk.Label(info, text=f"Duration: {self.duration:.2f} s").pack(side=tk.LEFT, padx=6)

        # preview frames
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

        # sliders
        sliders = ttk.Frame(self.top)
        sliders.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(sliders, text="Start (s)").grid(row=0, column=0, sticky='w', padx=4)
        self.start_scale = ttk.Scale(sliders, from_=0.0, to=self.duration, variable=self.start_sec, orient='horizontal', length=600, command=lambda v: self._on_start_change(v))
        self.start_scale.grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(sliders, text="End (s)").grid(row=1, column=0, sticky='w', padx=4)
        self.end_scale = ttk.Scale(sliders, from_=0.0, to=self.duration, variable=self.end_sec, orient='horizontal', length=600, command=lambda v: self._on_end_change(v))
        self.end_scale.grid(row=1, column=1, padx=4, pady=4)

        # bottom buttons
        bottom_info = ttk.Frame(self.top)
        bottom_info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.range_label = ttk.Label(bottom_info, text="")
        self.range_label.pack(side=tk.LEFT, padx=6)
        ttk.Button(bottom_info, text="Fine +0.01s Start", command=lambda: self._fine_adjust(0.01, True)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom_info, text="Fine -0.01s Start", command=lambda: self._fine_adjust(-0.01, True)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom_info, text="Fine +0.01s End", command=lambda: self._fine_adjust(0.01, False)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom_info, text="Fine -0.01s End", command=lambda: self._fine_adjust(-0.01, False)).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bottom_info, text="OK", command=self._on_ok).pack(side=tk.RIGHT, padx=6)
        ttk.Button(bottom_info, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=6)

        self._update_range_label()

    def _fine_adjust(self, delta, start=True):
        if start:
            self.start_sec.set(max(0.0, min(self.duration, self.start_sec.get() + delta)))
        else:
            self.end_sec.set(max(0.0, min(self.duration, self.end_sec.get() + delta)))
        self._update_range_label()
        self._update_previews()

    def _on_start_change(self, val):
        if self.start_sec.get() > self.end_sec.get():
            self.end_sec.set(self.start_sec.get())
        self._update_range_label()
        self._update_previews()

    def _on_end_change(self, val):
        if self.end_sec.get() < self.start_sec.get():
            self.start_sec.set(self.end_sec.get())
        self._update_range_label()
        self._update_previews()

    def _update_range_label(self):
        s,e = self.start_sec.get(), self.end_sec.get()
        start_frame = int(round(s*self.fps))
        end_frame = int(round(e*self.fps))
        length_frames = max(0, end_frame - start_frame + 1)
        self.range_label.config(text=f"Selected: {s:.3f}s → {e:.3f}s | Frames: {start_frame} → {end_frame} ({length_frames})")

    def _update_previews(self):
        start_idx = int(round(self.start_sec.get()*self.fps))
        end_idx = int(round(self.end_sec.get()*self.fps))
        frame_s = self.preview_callback(start_idx)
        frame_e = self.preview_callback(end_idx)
        self._display_in_frame(self.left_canvas, frame_s, f"Start #{start_idx}")
        self._display_in_frame(self.right_canvas, frame_e, f"End #{end_idx}")

    def _display_in_frame(self, container, frame_bgr, title="Preview"):
        for w in container.winfo_children():
            w.destroy()
        if frame_bgr is None:
            ttk.Label(container, text="(frame unavailable)").pack(expand=True)
            return
        gray = gray_if_needed(frame_bgr)
        h,w = gray.shape[:2]; side=min(h,w); cx,cy=w//2,h//2
        crop = gray[cy-side//2:cy+side//2, cx-side//2:cx+side//2]
        thumb = cv2.resize(crop, (420,420), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
        fig = plt.Figure(figsize=(3.5,3.5))
        ax = fig.add_subplot(111)
        ax.imshow(rgb, cmap='gray'); ax.axis('off'); ax.set_title(title)
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_ok(self):
        self.result = (self.start_sec.get(), self.end_sec.get())
        self.top.destroy()
    def _on_cancel(self):
        self.result = None
        self.top.destroy()

# -------------------------
# Main App
# -------------------------
class VideoPreprocessorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Preprocessor")
        self.video_path = None
        self.cap = None
        self.first_frame = None
        self.total_frames = 0
        self.fps = 30.0
        self.square_pts = None
        self.square_bbox = None
        self.start_sec = 0.0
        self.end_sec = 0.0
        self.alpha = tk.DoubleVar(value=1.0)
        self.beta = tk.IntVar(value=0)
        self.threshold = tk.IntVar(value=127)
        self.invert_lut = tk.BooleanVar(value=False)
        self._export_thread = None
        self._export_cancel = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------------
    # _build_ui (full)
    # -------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Button(top, text="Open Video", command=self.open_video).pack(side=tk.LEFT)
        self.open_label = ttk.Label(top, text="No video loaded")
        self.open_label.pack(side=tk.LEFT, padx=12)
        self.roi_btn = ttk.Button(top, text="Edit Square ROI", command=self.edit_roi, state='disabled')
        self.roi_btn.pack(side=tk.LEFT, padx=6)
        self.trim_btn = ttk.Button(top, text="Trim (seconds)...", command=self.open_trim_dialog, state='disabled')
        self.trim_btn.pack(side=tk.LEFT, padx=6)

        params = ttk.Frame(self.root)
        params.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(params, text="Contrast").grid(row=0,column=0,sticky='w')
        ttk.Scale(params, from_=0.2, to=3.0, variable=self.alpha, orient='horizontal', length=220, command=lambda v:self.preview()).grid(row=0,column=1)
        ttk.Label(params, text="Brightness").grid(row=0,column=2,sticky='w')
        ttk.Scale(params, from_=-100, to=100, variable=self.beta, orient='horizontal', length=220, command=lambda v:self.preview()).grid(row=0,column=3)
        ttk.Label(params, text="Threshold").grid(row=0,column=4,sticky='w')
        ttk.Scale(params, from_=0, to=255, variable=self.threshold, orient='horizontal', length=220, command=lambda v:self.preview()).grid(row=0,column=5)
        ttk.Checkbutton(params, text="Invert LUT", variable=self.invert_lut, command=self.preview).grid(row=0,column=6,padx=12)

        actions = ttk.Frame(self.root)
        actions.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.preview_btn = ttk.Button(actions, text="Preview first frame", command=self.preview, state='disabled')
        self.preview_btn.pack(side=tk.LEFT)
        self.export_btn = ttk.Button(actions, text="Export TIFF (trimmed)", command=self.export_tiff, state='disabled')
        self.export_btn.pack(side=tk.LEFT)
        self.cancel_btn = ttk.Button(actions, text="Cancel Export", command=self.cancel_export, state='disabled')
        self.cancel_btn.pack(side=tk.LEFT)

        self.preview_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # -------------------------
    # Video methods
    # -------------------------
    def open_video(self):
        path = filedialog.askopenfilename(title="Select video file", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return
        try:
            cap = open_video_capture(path)
            ret, frame = cap.read()
            if not ret:
                raise IOError("Cannot read first frame")
            self.video_path = path
            self.cap = cap
            self.first_frame = frame.copy()
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.start_sec = 0.0
            self.end_sec = self.total_frames/self.fps
            self.open_label.config(text=os.path.basename(path))
            self.roi_btn.config(state='normal')
            self.trim_btn.config(state='normal')
            self.preview_btn.config(state='normal')
            self.export_btn.config(state='normal')
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open video: {e}")

    def edit_roi(self):
        if self.first_frame is None:
            return
        editor = SquareROIEditor(self.first_frame)
        res = editor.run()
        if res is not None:
            self.square_pts = res
            xs = res[:,0]; ys = res[:,1]
            self.square_bbox = (min(xs), min(ys), max(xs), max(ys))
            self.preview()

    def open_trim_dialog(self):
        if self.cap is None:
            return
        dialog = TrimDialog(self.root, self.total_frames, self.fps, self.start_sec, self.end_sec, self._get_frame_by_index)
        self.root.wait_window(dialog.top)
        if dialog.result:
            self.start_sec, self.end_sec = dialog.result
            self.preview()

    def _get_frame_by_index(self, idx):
        if self.cap is None:
            return None
        idx = int(max(0,min(idx,self.total_frames-1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _apply_preview(self, frame):
        if frame is None:
            return None
        frame = gray_if_needed(frame)
        a = self.alpha.get(); b = self.beta.get()
        frame = cv2.convertScaleAbs(frame, alpha=a, beta=b)
        _, frame = cv2.threshold(frame, self.threshold.get(), 255, cv2.THRESH_BINARY)
        if self.invert_lut.get():
            frame = cv2.bitwise_not(frame)
        if self.square_bbox:
            x1,y1,x2,y2 = self.square_bbox
            frame = frame[y1:y2, x1:x2]
        frame = resize_1080(frame)
        return frame

    def preview(self):
        frame = self._apply_preview(self.first_frame)
        for w in self.preview_frame.winfo_children():
            w.destroy()
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        fig = plt.Figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.imshow(rgb, cmap='gray'); ax.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------------------
    # Export
    # -------------------------
    def export_tiff(self):
        if self.cap is None:
            return
        save_path = filedialog.asksaveasfilename(title="Save as TIFF", defaultextension=".tif", filetypes=[("TIFF", "*.tif")])
        if not save_path:
            return
        self._export_thread = threading.Thread(target=self._export_tiff_thread, args=(save_path,))
        self._export_cancel = False
        self.cancel_btn.config(state='normal')
        self._export_thread.start()

    def cancel_export(self):
        self._export_cancel = True

    def _export_tiff_thread(self, save_path):
        try:
            start_frame = int(round(self.start_sec*self.fps))
            end_frame = int(round(self.end_sec*self.fps))
            cap = open_video_capture(self.video_path)
            frames_out = []
            for fidx in range(start_frame, end_frame+1):
                if self._export_cancel:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_proc = self._apply_preview(frame)
                frames_out.append(frame_proc)
            if frames_out:
                tifffile.imwrite(save_path, np.stack(frames_out))
            messagebox.showinfo("Done", f"TIFF saved: {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
        finally:
            self.cancel_btn.config(state='disabled')

    # -------------------------
    # Close
    # -------------------------
    def _on_close(self):
        safe_destroy_opencv()
        self.root.destroy()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPreprocessorApp(root)
    root.mainloop()
