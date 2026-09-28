import os
import re
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

BASE_DIR = r"C:\Users\shrama\Desktop\Dam updated dataset\Dataset"
IMAGE_DIR = os.path.join(BASE_DIR, "BGSL_ROI")
OUTPUT_DIR = os.path.join(BASE_DIR, "Stage22_Rolling_Reference_Validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_PAIRS = [(1274,1374),(2000,2100),(3000,3100),(6400,6500)]
SUBSET_SIZE=41
SEARCH_SIZE=81
STEP=10
VARIANCE_THRESHOLD=500
ZNCC_THRESHOLD=0.98
MAX_PIXEL_DISPLACEMENT=20
GRID_STEP=10
GAUSSIAN_SIGMA=2.0
FIXTURE_X1,FIXTURE_X2,FIXTURE_Y1,FIXTURE_Y2=930,1080,450,638

def get_frame_number(path):
    nums=re.findall(r"\d+",os.path.basename(path))
    return int(nums[-2]) if len(nums)>=2 else None

def build_image_index():
    idx={}
    files=glob.glob(os.path.join(IMAGE_DIR,"*.tif"))+glob.glob(os.path.join(IMAGE_DIR,"*.tiff"))
    for p in files:
        f=get_frame_number(p)
        if f is not None and f!=0: idx[f]=p
    return idx

def subpixel_offset(left,center,right):
    d=left-2.0*center+right
    return 0.0 if abs(d)<1e-10 else 0.5*(left-right)/d

def track_pair_fast(img1,img2):
    hs=SUBSET_SIZE//2; hh=SEARCH_SIZE//2; margin=hs+hh; out=[]
    for cy in range(margin,img1.shape[0]-margin,STEP):
        for cx in range(margin,img1.shape[1]-margin,STEP):
            template=img1[cy-hs:cy+hs+1,cx-hs:cx+hs+1]
            if template.shape!=(SUBSET_SIZE,SUBSET_SIZE) or np.var(template)<VARIANCE_THRESHOLD: continue
            top,left=cy-hh,cx-hh
            sr=img2[top:cy+hh+1,left:cx+hh+1]
            if sr.shape!=(SEARCH_SIZE,SEARCH_SIZE): continue
            corr=cv2.matchTemplate(sr,template,cv2.TM_CCOEFF_NORMED)
            _,score,_,loc=cv2.minMaxLoc(corr)
            if np.isnan(score) or score<ZNCC_THRESHOLD: continue
            dx=left+loc[0]-(cx-hs); dy=top+loc[1]-(cy-hs)
            if abs(dx)>MAX_PIXEL_DISPLACEMENT or abs(dy)>MAX_PIXEL_DISPLACEMENT: continue
            mx,my=loc
            c=corr[my,mx]
            l=corr[my,mx-1] if mx>0 else c; r=corr[my,mx+1] if mx<corr.shape[1]-1 else c
            u=corr[my-1,mx] if my>0 else c; d=corr[my+1,mx] if my<corr.shape[0]-1 else c
            dx+=subpixel_offset(l,c,r); dy+=subpixel_offset(u,c,d)
            out.append([cx,cy,dx,dy,float(score)])
    return out

def calculate_strain(results):
    if len(results)<20:return None
    df=pd.DataFrame(results,columns=["X","Y","dx","dy","ZNCC"]).replace([np.inf,-np.inf],np.nan).dropna()
    if len(df)<20:return None
    x,y,dx,dy=[df[c].to_numpy(float) for c in ["X","Y","dx","dy"]]
    gx=np.arange(x.min(),x.max()+GRID_STEP,GRID_STEP); gy=np.arange(y.min(),y.max()+GRID_STEP,GRID_STEP)
    X,Y=np.meshgrid(gx,gy); pts=np.column_stack([x,y])
    gdx=griddata(pts,dx,(X,Y),method="linear"); gdy=griddata(pts,dy,(X,Y),method="linear")
    ndx=griddata(pts,dx,(X,Y),method="nearest"); ndy=griddata(pts,dy,(X,Y),method="nearest")
    gdx[~np.isfinite(gdx)]=ndx[~np.isfinite(gdx)]; gdy[~np.isfinite(gdy)]=ndy[~np.isfinite(gdy)]
    gdx=gaussian_filter(gdx,sigma=GAUSSIAN_SIGMA); gdy=gaussian_filter(gdy,sigma=GAUSSIAN_SIGMA)
    du_dy,du_dx=np.gradient(gdx,GRID_STEP,GRID_STEP); dv_dy,dv_dx=np.gradient(gdy,GRID_STEP,GRID_STEP)
    exx=du_dx; eyy=dv_dy; exy=0.5*(du_dy+dv_dx)
    principal=0.5*(exx+eyy)+np.sqrt(((exx-eyy)*0.5)**2+exy**2)
    return X,Y,principal

def save_heatmap(frame1,frame2,img,X,Y,P,path):
    vals=P[np.isfinite(P)]; vals=np.maximum(vals,0)
    vmax=np.percentile(vals,99) if len(vals) else 1.0
    if vmax<=0:vmax=max(float(vals.max()) if len(vals) else 0,1.0)
    fig,ax=plt.subplots(figsize=(14,8))
    h=ax.imshow(np.maximum(P,0),extent=[X.min(),X.max(),Y.max(),Y.min()],cmap="turbo",vmin=0,vmax=vmax,alpha=.78,interpolation="nearest")
    ax.imshow(img,cmap="gray",extent=[0,img.shape[1],img.shape[0],0],alpha=.28)
    ax.set_title(f"Rolling-Reference DIC Principal Strain — {frame1} → {frame2}")
    ax.set_xlabel("Image X (pixels)"); ax.set_ylabel("Image Y (pixels)")
    fig.colorbar(h,ax=ax,label="Principal strain")
    plt.tight_layout(); plt.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)

def analyze(frame1,frame2,idx):
    print("\n"+"="*65); print(f"VALIDATION: {frame1} -> {frame2}"); print("="*65)
    p1,p2=idx.get(frame1),idx.get(frame2)
    if not p1 or not p2: print("SKIPPED: one or both frames are unavailable."); return None
    img1=cv2.imread(p1,cv2.IMREAD_GRAYSCALE); img2=cv2.imread(p2,cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None: print("SKIPPED: image could not be loaded."); return None
    print("Reference:",p1); print("Target:",p2); print("Running rolling-reference DIC...")
    res=track_pair_fast(img1,img2); print("Tracked points:",len(res))
    if not res:return None
    df=pd.DataFrame(res,columns=["X","Y","dx","dy","ZNCC"]); mag=np.hypot(df.dx,df.dy)
    print(f"Average ZNCC: {df.ZNCC.mean():.6f}"); print(f"Average DIC magnitude: {mag.mean():.6f}"); print(f"Maximum DIC magnitude: {mag.max():.6f}")
    strain=calculate_strain(res)
    if strain is None: print("SKIPPED: insufficient strain data."); return None
    X,Y,P=strain; mask=(X>=FIXTURE_X1)&(X<=FIXTURE_X2)&(Y>=FIXTURE_Y1)&(Y<=FIXTURE_Y2)
    vals=np.maximum(P[(~mask)&np.isfinite(P)],0)
    if not len(vals):return None
    mean,p95,p99,mx=[float(np.mean(vals)),float(np.percentile(vals,95)),float(np.percentile(vals,99)),float(np.max(vals))]
    A=P.copy(); A[mask]=np.nan; safe=np.where(np.isfinite(A),A,-np.inf); iy,ix=np.unravel_index(np.argmax(safe),safe.shape)
    maxx,maxy=float(X[iy,ix]),float(Y[iy,ix])
    print("\nSTRAIN"); print(f"Mean : {mean:.6f}"); print(f"P95  : {p95:.6f}"); print(f"P99  : {p99:.6f}"); print(f"Max  : {mx:.6f}")
    print("\nMAX STRAIN LOCATION"); print(f"X : {maxx:.1f}"); print(f"Y : {maxy:.1f}")
    hp=os.path.join(OUTPUT_DIR,f"rolling_heatmap_{frame1}_{frame2}.png"); save_heatmap(frame1,frame2,img2,X,Y,P,hp); print("Saved:",hp)
    sp=os.path.join(OUTPUT_DIR,f"strain_field_{frame1}_{frame2}.csv"); pd.DataFrame({"X":X.ravel(),"Y":Y.ravel(),"Principal_Strain":P.ravel()}).to_csv(sp,index=False)
    return {"Frame_1":frame1,"Frame_2":frame2,"Tracked_Points":len(res),"Average_ZNCC":df.ZNCC.mean(),"Average_DIC_Magnitude":mag.mean(),"Maximum_DIC_Magnitude":mag.max(),"Principal_Strain_Mean":mean,"Principal_Strain_P95":p95,"Principal_Strain_P99":p99,"Principal_Strain_Max":mx,"Max_Strain_X":maxx,"Max_Strain_Y":maxy,"Heatmap_File":hp}

def main():
    print("="*75); print("STAGE 22 - ROLLING-REFERENCE HEATMAP VALIDATION"); print("="*75)
    idx=build_image_index(); print(f"Available indexed images: {len(idx)}")
    results=[]
    for a,b in TEST_PAIRS:
        r=analyze(a,b,idx)
        if r:results.append(r)
    if results:
        out=os.path.join(OUTPUT_DIR,"rolling_reference_validation_summary.csv"); pd.DataFrame(results).to_csv(out,index=False)
        print("\n"+"="*75); print("STAGE 22 COMPLETE"); print("="*75); print(f"Successful validations : {len(results)}"); print("Summary:",out); print("Review all heatmaps before final integration.")
    else: print("No validation results were produced.")

if __name__=="__main__":main()