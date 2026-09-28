import os, re, glob, math, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

# IMPORTANT:
# This version uses streamlit-cropper for the ROI.
# It gives the old-style drag-a-rectangle interface.
try:
    from streamlit_cropper import st_cropper
    CROPPER_OK = True
except Exception:
    CROPPER_OK = False

st.set_page_config(page_title="DIC Concrete Beam Analyzer", page_icon="📐", layout="wide")
st.title("📐 DIC Concrete Beam Analysis System")
st.caption("Image + Excel synchronization • Manual beam ROI • Fast DIC • Strain heatmap • Experimental graphs")

# ----------------------------- helpers -----------------------------

def frame_number(filename):
    nums = re.findall(r"\d+", Path(filename).name)
    if len(nums) >= 2:
        return int(nums[-2])
    return int(nums[-1]) if nums else None

@st.cache_data(show_spinner=False)
def build_frame_index(folder):
    files = glob.glob(str(Path(folder) / "*.tif")) + glob.glob(str(Path(folder) / "*.TIF"))
    out = {}
    for f in files:
        n = frame_number(f)
        if n is not None and n != 0:
            out[n] = f
    return dict(sorted(out.items()))

def read_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Could not read {path}")
    return im

def read_rgb(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Could not read {path}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def parse_excel(source):
    df = pd.read_excel(source)
    required = ["Image Number", "Time_0", "Load ch 2 kN", "Displacement ch 7 µm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing Excel columns: " + ", ".join(missing))
    df = df.copy()
    df["Image Number"] = pd.to_numeric(df["Image Number"], errors="coerce")
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Image Number"])
    df["Image Number"] = df["Image Number"].astype(int)
    return df.sort_values("Image Number").drop_duplicates("Image Number")

def fast_dic(ref, tar, subset=41, search=81, step=10, threshold=.98):
    if subset % 2 == 0: subset += 1
    if search % 2 == 0: search += 1
    r, sr = subset // 2, search // 2
    h, w = ref.shape
    rows = []

    for cy in range(sr, h-sr, step):
        for cx in range(sr, w-sr, step):
            template = ref[cy-r:cy+r+1, cx-r:cx+r+1]
            if template.size == 0 or np.var(template) < 500:
                continue
            search_img = tar[cy-sr:cy+sr+1, cx-sr:cx+sr+1]
            result = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if not np.isfinite(score) or score < threshold:
                continue

            dx = float(loc[0] - (sr-r))
            dy = float(loc[1] - (sr-r))

            if abs(dx) > 5 or abs(dy) > 5:
                continue
            rows.append((cx, cy, dx, dy, float(score)))

    return pd.DataFrame(rows, columns=["X","Y","dx","dy","ZNCC"])

def local_strain(dic, neighbors=25):
    d = dic.replace([np.inf,-np.inf],np.nan).dropna()
    d = d[d.ZNCC >= .95].copy()
    d = d.groupby(["X","Y"],as_index=False).agg({"dx":"median","dy":"median","ZNCC":"median"})
    if len(d) < neighbors:
        raise ValueError("Not enough valid DIC points.")

    pts=d[["X","Y"]].to_numpy(float)
    u=d.dx.to_numpy(float); v=d.dy.to_numpy(float)
    tree=cKDTree(pts)
    rows=[]

    for i,p0 in enumerate(pts):
        dist,idx=tree.query(p0,k=min(neighbors,len(d)))
        if np.isscalar(idx):
            idx=np.array([idx]); dist=np.array([dist])
        x=pts[idx,0]-p0[0]; y=pts[idx,1]-p0[1]
        A=np.column_stack([np.ones(len(x)),x,y])
        w=1/(dist+2)
        W=np.diag(w)
        try:
            au=np.linalg.lstsq(W@A,W@u[idx],rcond=None)[0]
            av=np.linalg.lstsq(W@A,W@v[idx],rcond=None)[0]
            exx, eyy = au[1], av[2]
            exy=.5*(au[2]+av[1])
            avg=.5*(exx+eyy)
            rad=np.sqrt(max(0,((exx-eyy)*.5)**2+exy**2))
            rows.append((p0[0],p0[1],exx,eyy,exy,avg+rad,avg-rad))
        except Exception:
            pass

    return pd.DataFrame(rows,columns=["X","Y","exx","eyy","exy","Principal_1","Principal_2"])

def heatmap(image, strain):
    h,w=image.shape[:2]
    z=strain.Principal_1.to_numpy(float)
    x=strain.X.to_numpy(float); y=strain.Y.to_numpy(float)
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(z)
    x,y,z=x[ok],y[ok],z[ok]
    gx=np.arange(0,w,5); gy=np.arange(0,h,5)
    GX,GY=np.meshgrid(gx,gy)
    Z=griddata((x,y),z,(GX,GY),method="linear")
    Zn=griddata((x,y),z,(GX,GY),method="nearest")
    Z=np.where(np.isfinite(Z),Z,Zn)
    Z=gaussian_filter(Z,1.2)
    lo=float(np.percentile(z,5)); hi=float(np.percentile(z,99))
    if hi<=lo: hi=lo+1e-6

    fig,ax=plt.subplots(figsize=(14,5.5))
    ax.imshow(image,cmap="gray")
    im=ax.imshow(Z,extent=[0,w,h,0],cmap="turbo",alpha=.72,vmin=lo,vmax=hi)
    ax.set_title("DIC Local-Affine Principal Strain Heatmap")
    ax.set_xlabel("X (pixels)"); ax.set_ylabel("Y (pixels)")
    plt.colorbar(im,ax=ax,label="Principal strain")
    plt.tight_layout()
    return fig, z

def event_screen(df):
    d=df.sort_values("Image Number").copy()
    d["dLoad"]=d["Load ch 2 kN"].diff()
    d["dDisp"]=d["Displacement ch 7 µm"].diff()
    lt=float(d.dLoad.abs().quantile(.95))
    dt=float(d.dDisp.abs().quantile(.95))
    ev=d[(d.dLoad < -lt)&(d.dDisp > dt)].copy()
    return ev,lt,dt

# ----------------------------- sidebar -----------------------------

st.sidebar.header("Dataset")
root=st.sidebar.text_input("Project folder",r"C:\Users\shrama\Desktop\Dam updated dataset")
img_folder=st.sidebar.text_input("Image folder",str(Path(root)/"Dataset"/"BGSL"))
excel_path=st.sidebar.text_input("Excel file",str(Path(root)/"experimental_data"/"combined_image_load_data_0_to_6600.xlsx"))
upload=st.sidebar.file_uploader("Or upload Excel",type=["xlsx","xls"])

st.sidebar.header("DIC")
subset=st.sidebar.slider("Subset size",21,61,41,2)
search=st.sidebar.slider("Search window",41,101,81,2)
step=st.sidebar.slider("Grid step",5,20,10)
threshold=st.sidebar.slider("ZNCC threshold",.90,.995,.98,.005)

if not CROPPER_OK:
    st.error("The drag-to-select ROI component is not installed.")
    st.code("python -m pip install streamlit-cropper",language="bash")
    st.info("Install it once, then restart the dashboard.")
    st.stop()

if not Path(img_folder).exists():
    st.warning("Image folder not found. Enter the correct local BGSL folder.")
    st.stop()

frames=build_frame_index(img_folder)
if not frames:
    st.error("No TIFF images found.")
    st.stop()

try:
    exp=parse_excel(upload if upload is not None else excel_path) if (upload is not None or Path(excel_path).exists()) else None
except Exception as e:
    st.error(str(e)); st.stop()

# ----------------------------- overview -----------------------------

st.subheader("Dataset overview")
a,b,c,d=st.columns(4)
a.metric("Available images",f"{len(frames):,}")
b.metric("First frame",min(frames))
c.metric("Last frame",max(frames))
d.metric("Excel rows",f"{len(exp):,}" if exp is not None else "Not loaded")

# ----------------------------- ROI -----------------------------

st.header("1. Select Beam ROI")

frame_list=list(frames)
ref_frame=st.selectbox("Reference image",frame_list)
ref_path=frames[ref_frame]
ref_rgb=read_rgb(ref_path)

st.write(f"Reference: `{Path(ref_path).name}` — {ref_rgb.shape[1]} × {ref_rgb.shape[0]} pixels")
st.info("Drag a rectangle over the concrete beam region. This is the same manual ROI style as the earlier version.")

# cropper uses the image itself and returns the cropped PIL image.
# realtime_update=False means the crop is finalized by the widget control.
cropped = st_cropper(
    Image.fromarray(ref_rgb),
    realtime_update=True,
    box_color="#ff0000",
    aspect_ratio=None,
    return_type="box",
)

if cropped is not None:
    # streamlit-cropper returns a dictionary for return_type="box"
    x = int(cropped["left"])
    y = int(cropped["top"])
    w = int(cropped["width"])
    h = int(cropped["height"])
    if w>1 and h>1:
        roi=(x,y,w,h)
        st.session_state["roi"]=roi
        preview=ref_rgb.copy()
        cv2.rectangle(preview,(x,y),(x+w,y+h),(255,0,0),4)
        st.image(preview,caption=f"Selected ROI: x={x}, y={y}, width={w}, height={h}")
        st.success("ROI selected. You can adjust the rectangle directly above.")

roi=st.session_state.get("roi")
if roi is None:
    st.stop()

# ----------------------------- DIC -----------------------------

st.header("2. Run DIC")

targets=[f for f in frame_list if f!=ref_frame]
if not targets:
    st.error("No target frames available.")
    st.stop()

target_frame=st.selectbox("Target image",targets)

if st.button("🚀 Run DIC + Strain Heatmap",type="primary"):
    rx,ry,rw,rh=roi
    with st.spinner("Running fast DIC inside selected ROI..."):
        t0=time.time()
        ref_gray=read_gray(ref_path)[ry:ry+rh,rx:rx+rw]
        tar_gray=read_gray(frames[target_frame])[ry:ry+rh,rx:rx+rw]
        dic=fast_dic(ref_gray,tar_gray,subset,search,step,threshold)
        elapsed=time.time()-t0

    if dic.empty:
        st.error("No valid DIC points. Try a better textured ROI or slightly lower ZNCC.")
        st.stop()

    st.session_state["dic"]=dic
    st.session_state["dic_time"]=elapsed
    st.session_state["dic_ref"]=ref_frame
    st.session_state["dic_target"]=target_frame

dic=st.session_state.get("dic")
if dic is not None:
    m1,m2,m3,m4=st.columns(4)
    mag=np.sqrt(dic.dx**2+dic.dy**2)
    m1.metric("Tracked points",f"{len(dic):,}")
    m2.metric("Mean ZNCC",f"{dic.ZNCC.mean():.5f}")
    m3.metric("Mean displacement",f"{mag.mean():.4f} px")
    m4.metric("Maximum displacement",f"{mag.max():.4f} px")
    st.caption(f"DIC processing time: {st.session_state['dic_time']:.2f} seconds")

    st.header("3. DIC Strain Heatmap")
    try:
        strain=local_strain(dic)
        rx,ry,rw,rh=roi
        roi_rgb=ref_rgb[ry:ry+rh,rx:rx+rw]
        fig,z=heatmap(roi_rgb,strain)
        st.pyplot(fig,clear_figure=True)

        h1,h2,h3,h4=st.columns(4)
        h1.metric("Mean strain",f"{np.mean(z):.6f}")
        h2.metric("P95",f"{np.percentile(z,95):.6f}")
        h3.metric("P99",f"{np.percentile(z,99):.6f}")
        h4.metric("Maximum",f"{np.max(z):.6f}")

        strain_out=strain.copy()
        strain_out["X_full"]=strain_out.X+rx
        strain_out["Y_full"]=strain_out.Y+ry
        st.download_button(
            "⬇️ Download strain CSV",
            strain_out.to_csv(index=False).encode(),
            f"strain_{ref_frame}_{target_frame}.csv",
            "text/csv"
        )
    except Exception as e:
        st.error(f"Heatmap generation failed: {e}")

# ----------------------------- graphs -----------------------------

st.header("4. Experimental Graphs")
if exp is None:
    st.info("Load the Excel file to display synchronized experimental graphs.")
else:
    tabs=st.tabs(["Load vs Time","Displacement vs Time","Load vs Displacement"])
    with tabs[0]:
        fig,ax=plt.subplots(figsize=(10,4.5))
        ax.plot(exp.Time_0,exp["Load ch 2 kN"])
        ax.set_xlabel("Time"); ax.set_ylabel("Load (kN)")
        ax.set_title("Load vs Time"); ax.grid(alpha=.25)
        st.pyplot(fig,clear_figure=True)
    with tabs[1]:
        fig,ax=plt.subplots(figsize=(10,4.5))
        ax.plot(exp.Time_0,exp["Displacement ch 7 µm"])
        ax.set_xlabel("Time"); ax.set_ylabel("Displacement ch 7 (µm)")
        ax.set_title("Displacement vs Time"); ax.grid(alpha=.25)
        st.pyplot(fig,clear_figure=True)
    with tabs[2]:
        fig,ax=plt.subplots(figsize=(10,4.5))
        ax.plot(exp["Displacement ch 7 µm"],exp["Load ch 2 kN"])
        ax.set_xlabel("Displacement ch 7 (µm)"); ax.set_ylabel("Load (kN)")
        ax.set_title("Load vs Displacement"); ax.grid(alpha=.25)
        st.pyplot(fig,clear_figure=True)

# ----------------------------- events -----------------------------

st.header("5. Mechanical Event Screening")
if exp is not None:
    ev,lt,dt=event_screen(exp)
    e1,e2=st.columns(2)
    e1.metric("95th |ΔLoad| threshold",f"{lt:.4f} kN")
    e2.metric("95th |ΔDisplacement| threshold",f"{dt:.4f} µm")
    if len(ev):
        cols=[c for c in ["Image Number","Time_0","Load ch 2 kN","Displacement ch 7 µm","dLoad","dDisp"] if c in ev]
        st.dataframe(ev[cols].head(100),use_container_width=True)
    else:
        st.info("No mechanical transition candidates crossed the current thresholds.")

# ----------------------------- previous results -----------------------------

st.header("6. Previously Validated DIC Events")
stage17=Path(root)/"Dataset"/"Stage17_Final_Event_Synthesis"/"final_event_synthesis.csv"
if stage17.exists():
    try:
        old=pd.read_csv(stage17)
        st.dataframe(old,use_container_width=True)
        st.caption("These are the previously validated project-level event results. A localized deformation event is not automatically a confirmed crack.")
    except Exception as e:
        st.warning(str(e))
else:
    st.info("Stage 17 event synthesis was not found in the selected project folder.")

st.divider()
st.caption("DIC Concrete Beam Analysis • Fast OpenCV DIC • Manual drag ROI • Local-affine strain visualization")
