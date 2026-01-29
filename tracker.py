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
        self.fps = float(fps) if fps and fps>0 else 1.0
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
        info = ttk.Frame(self.top)
        info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(info, text=f"Total frames: {self.total_frames}").pack(side=tk.LEFT, padx=6)
        ttk.Label(info, text=f"FPS: {self.fps:.3f}").pack(side=tk.LEFT, padx=6)
        ttk.Label(info, text=f"Duration: {self.duration:.2f} s").pack(side=tk.LEFT, padx=6)

        # preview frames container
        pv = ttk.Frame(self.top)
        pv.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)
        left = ttk.LabelFrame(pv, text="Start Frame Preview")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=4)
        right = ttk.LabelFrame(pv, text="End Frame Preview")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.left_canvas = ttk.Frame(left); self.left_canvas.pack(fill=tk.BOTH, expand=True)
        self.right_canvas = ttk.Frame(right); self.right_canvas.pack(fill=tk.BOTH, expand=True)

        # sliders
        sliders = ttk.Frame(self.top)
        sliders.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(sliders, text="Start (s)").grid(row=0, column=0, sticky='w', padx=4)
        self.start_scale = ttk.Scale(sliders, from_=0.0, to=self.duration, variable=self.start_sec, orient='horizontal', length=600, command=self._on_start_change)
        self.start_scale.grid(row=0, column=1, padx=4, pady=4)
        ttk.Label(sliders, text="End (s)").grid(row=1, column=0, sticky='w', padx=4)
        self.end_scale = ttk.Scale(sliders, from_=0.0, to=self.duration, variable=self.end_sec, orient='horizontal', length=600, command=self._on_end_change)
        self.end_scale.grid(row=1, column=1, padx=4, pady=4)

        # labels and buttons
        bottom_info = ttk.Frame(self.top)
        bottom_info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.range_label = ttk.Label(bottom_info, text=""); self.range_label.pack(side=tk.LEFT, padx=6)
        ttk.Button(bottom_info, text="OK", command=self._on_ok).pack(side=tk.RIGHT, padx=6)
        ttk.Button(bottom_info, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT, padx=6)
        self._update_range_label()

    def _on_start_change(self, val): self._clamp_and_update(start=True)
    def _on_end_change(self, val): self._clamp_and_update(start=False)

    def _clamp_and_update(self, start=True):
        s = float(self.start_sec.get()); e = float(self.end_sec.get())
        if start and s>e: self.end_sec.set(s)
        if not start and e<s: self.start_sec.set(e)
        self._update_range_label(); self._update_previews()

    def _update_range_label(self):
        s = float(self.start_sec.get()); e = float(self.end_sec.get())
        start_frame = min(self.total_frames-1, max(0, int(round(s * self.fps))))
        end_frame = min(self.total_frames-1, max(0, int(round(e * self.fps))))
        length_frames = max(0, end_frame - start_frame + 1)
        length_sec = (length_frames / self.fps) if self.fps>0 else 0.0
        self.range_label.config(text=f"Selected: {s:.3f}s → {e:.3f}s  |  Frames: {start_frame} → {end_frame}  ({length_frames} frames, {length_sec:.3f} s)")

    def _update_previews(self):
        s = float(self.start_sec.get()); e = float(self.end_sec.get())
        start_idx = min(self.total_frames-1, max(0, int(round(s * self.fps))))
        end_idx = min(self.total_frames-1, max(0, int(round(e * self.fps))))
        frame_s = frame_e = None
        try: frame_s = self.preview_callback(start_idx)
        except Exception: pass
        try: frame_e = self.preview_callback(end_idx)
        except Exception: pass
        self._display_in_frame(self.left_canvas, frame_s, f"Start frame #{start_idx}")
        self._display_in_frame(self.right_canvas, frame_e, f"End frame #{end_idx}")
        self._update_range_label()

    def _display_in_frame(self, container, frame_bgr, title):
        for w in container.winfo_children(): w.destroy()
        if frame_bgr is None: ttk.Label(container, text="(frame unavailable)").pack(expand=True); return
        frame = frame_bgr.copy(); gray = gray_if_needed(frame)
        h,w = gray.shape[:2]; side = min(h,w); cx,cy=w//2,h//2
        crop = gray[max(0,cy-side//2):cy+side//2, max(0,cx-side//2):cx+side//2]
        thumb = cv2.resize(crop,(420,420),interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
        fig = plt.Figure(figsize=(3.5,3.5)); ax = fig.add_subplot(111); ax.imshow(rgb,cmap='gray'); ax.axis('off'); ax.set_title(title)
        canvas = FigureCanvasTkAgg(fig, master=container); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_ok(self): self.result = (float(self.start_sec.get()), float(self.end_sec.get())); self.top.destroy()
    def _on_cancel(self): self.result = None; self.top.destroy()

# -------------------------
# Fine Trim Adjustment
# -------------------------
class FineTrimDialog:
    def __init__(self, parent, start_sec, end_sec, duration):
        self.result = None
        self.top = tk.Toplevel(parent)
        self.top.title("Fine Trim Adjustment (seconds)")
        self.top.transient(parent)
        self.top.grab_set()
        ttk.Label(self.top, text="Start (s)").grid(row=0,column=0,padx=6,pady=4)
        ttk.Label(self.top, text="End (s)").grid(row=1,column=0,padx=6,pady=4)
        self.start_var = tk.DoubleVar(value=start_sec)
        self.end_var = tk.DoubleVar(value=end_sec)
        self.start_spin = ttk.Spinbox(self.top, from_=0.0, to=duration, increment=0.01, textvariable=self.start_var, width=10)
        self.end_spin = ttk.Spinbox(self.top, from_=0.0, to=duration, increment=0.01, textvariable=self.end_var, width=10)
        self.start_spin.grid(row=0,column=1,padx=6,pady=4)
        self.end_spin.grid(row=1,column=1,padx=6,pady=4)
        ttk.Button(self.top, text="OK", command=self._on_ok).grid(row=2,column=0,padx=6,pady=6)
        ttk.Button(self.top, text="Cancel", command=self._on_cancel).grid(row=2,column=1,padx=6,pady=6)

    def _on_ok(self):
        s = float(self.start_var.get()); e = float(self.end_var.get())
        if s>e: messagebox.showwarning("Invalid", "Start > End"); return
        self.result = (s,e); self.top.destroy()
    def _on_cancel(self): self.result = None; self.top.destroy()

# -------------------------
# Main App
# -------------------------
class VideoPreprocessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Preprocessor - Square ROI, Trim (seconds), Threshold, Export TIFF")
        self.root.geometry("1200x820")
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.first_frame = None

        # ROI
        self.square_pts = None
        self.square_bbox = None

        # trim
        self.start_sec = 0.0
        self.end_sec = 0.0

        # processing
        self.alpha = tk.DoubleVar(value=1.0)
        self.beta = tk.IntVar(value=0)
        self.threshold = tk.IntVar(value=127)
        self.invert_lut = tk.BooleanVar(value=False)

        # export
        self._export_thread = None
        self._export_cancel = False

        # build UI
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- BUILD UI ----------
    def _build_ui(self):
        top = ttk.Frame(self.root); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Button(top, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=6)
        self.open_label = ttk.Label(top, text="No video loaded"); self.open_label.pack(side=tk.LEFT, padx=12)
        self.roi_btn = ttk.Button(top, text="Edit Square ROI", command=self.edit_roi, state='disabled'); self.roi_btn.pack(side=tk.LEFT, padx=6)
        self.trim_btn = ttk.Button(top, text="Trim Video", command=self.trim_video, state='disabled'); self.trim_btn.pack(side=tk.LEFT, padx=6)
        self.fine_trim_btn = ttk.Button(top, text="Fine Trim Adjust", command=self.fine_trim_adjust, state='disabled'); self.fine_trim_btn.pack(side=tk.LEFT, padx=6)

        proc_frame = ttk.LabelFrame(self.root, text="Processing")
        proc_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        ttk.Label(proc_frame, text="Alpha (contrast)").grid(row=0,column=0,padx=4,pady=4)
        ttk.Scale(proc_frame, from_=0.1,to=3.0,variable=self.alpha,orient='horizontal',length=200).grid(row=0,column=1,padx=4,pady=4)
        ttk.Label(proc_frame, text="Beta (brightness)").grid(row=1,column=0,padx=4,pady=4)
        ttk.Scale(proc_frame, from_=-100,to=100,variable=self.beta,orient='horizontal',length=200).grid(row=1,column=1,padx=4,pady=4)
        ttk.Label(proc_frame, text="Threshold").grid(row=2,column=0,padx=4,pady=4)
        ttk.Scale(proc_frame, from_=0,to=255,variable=self.threshold,orient='horizontal',length=200).grid(row=2,column=1,padx=4,pady=4)
        ttk.Checkbutton(proc_frame, text="Invert LUT", variable=self.invert_lut).grid(row=3,column=0,columnspan=2,padx=4,pady=4)

        ttk.Button(self.root, text="Export as TIFF", command=self.export_tiff, state='disabled').pack(side=tk.TOP, pady=8)

    # ---------- Video operations ----------
    def open_video(self):
        path = filedialog.askopenfilename(title="Select video", filetypes=[("Video files","*.mp4 *.avi *.mov *.mkv")])
        if not path: return
        self.video_path = path
        self.cap = open_video_capture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.first_frame = None
        ret, frame = self.cap.read()
        if ret: self.first_frame = frame.copy()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        self.open_label.config(text=os.path.basename(path))
        self.roi_btn.config(state='normal')
        self.trim_btn.config(state='normal')
        self.fine_trim_btn.config(state='normal')

    def edit_roi(self):
        if self.first_frame is None: messagebox.showwarning("No Video","Load a video first"); return
        editor = SquareROIEditor(self.first_frame)
        pts = editor.run()
        if pts is not None:
            self.square_pts = pts
            x1 = np.min(pts[:,0]); y1=np.min(pts[:,1]); x2=np.max(pts[:,0]); y2=np.max(pts[:,1])
            self.square_bbox = (x1,y1,x2,y2)
            messagebox.showinfo("ROI set", f"Square ROI set: {self.square_bbox}")

    def trim_video(self):
        if not self.cap: return
        dlg = TrimDialog(self.root, self.total_frames, self.fps, self.start_sec, self.end_sec, self._get_frame_at)
        self.root.wait_window(dlg.top)
        if dlg.result:
            self.start_sec, self.end_sec = dlg.result
            messagebox.showinfo("Trim set", f"Trim range set: {self.start_sec:.3f}s → {self.end_sec:.3f}s")

    def fine_trim_adjust(self):
        dlg = FineTrimDialog(self.root, self.start_sec, self.end_sec, self.total_frames/self.fps)
        self.root.wait_window(dlg.top)
        if dlg.result:
            self.start_sec, self.end_sec = dlg.result
            messagebox.showinfo("Trim adjusted", f"Fine Trim range: {self.start_sec:.3f}s → {self.end_sec:.3f}s")

    def _get_frame_at(self, idx):
        if not self.cap: return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret: return None
        return frame

    def export_tiff(self):
        if not self.cap: messagebox.showwarning("No video","Load a video first"); return
        out_path = filedialog.asksaveasfilename(title="Save TIFF", defaultextension=".tif", filetypes=[("TIFF","*.tif")])
        if not out_path: return
        start_idx = int(round(self.start_sec*self.fps))
        end_idx = int(round(self.end_sec*self.fps))
        frames_to_export = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for fidx in range(start_idx, end_idx+1):
            ret, frame = self.cap.read()
            if not ret: break
            gray = gray_if_needed(frame)
            x1,y1,x2,y2 = self.square_bbox if self.square_bbox else (0,0,gray.shape[1], gray.shape[0])
            crop = gray[y1:y2, x1:x2]
            adj = cv2.convertScaleAbs(crop, alpha=self.alpha.get(), beta=self.beta.get())
            _, thresh = cv2.threshold(adj, self.threshold.get(), 255, cv2.THRESH_BINARY)
            if self.invert_lut.get(): thresh = cv2.bitwise_not(thresh)
            frames_to_export.append(thresh)
        if frames_to_export:
            tifffile.imwrite(out_path, np.array(frames_to_export))
            messagebox.showinfo("Exported", f"Exported {len(frames_to_export)} frames to {out_path}")

    def _on_close(self):
        safe_destroy_opencv()
        if self.cap: self.cap.release()
        self.root.destroy()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPreprocessorApp(root)
    root.mainloop()
