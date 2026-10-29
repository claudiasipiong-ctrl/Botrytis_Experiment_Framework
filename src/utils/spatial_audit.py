"""Auxiliary Grad-CAM spatial statistics. No class label is derived here."""
from __future__ import annotations
import numpy as np
from typing import Any, Sequence


def gradcam_sign_metrics(cam: np.ndarray, sign_boxes: Sequence[Sequence[float]], crop_box: Sequence[float]) -> dict[str,Any]:
    """Energy and peak inside the union of class 2 boxes intersecting the crop.

    Boxes are in the original-image coordinate frame. The crop box is the exact
    integer window used for cropping; the CAM is resized to its own grid. No
    annotation => NA, including positive fruits without class 2 regions.
    """
    heat = np.asarray(cam,dtype=float)
    if heat.ndim != 2 or not np.isfinite(heat).all() or (heat<0).any():
        raise ValueError("CAM must be a finite nonnegative 2D array")
    left,top,right,bottom=map(float,crop_box)
    if right<=left or bottom<=top:
        raise ValueError("Degenerate crop window")
    h,w=heat.shape
    xs=left+(np.arange(w)+.5)*(right-left)/w
    ys=top+(np.arange(h)+.5)*(bottom-top)/h
    mask=np.zeros((h,w),dtype=bool)
    present=0
    for x1,y1,x2,y2 in sign_boxes:
        if min(x2,right)<=max(x1,left) or min(y2,bottom)<=max(y1,top):continue
        present+=1
        mask |= (ys[:,None]>=y1)&(ys[:,None]<y2)&(xs[None,:]>=x1)&(xs[None,:]<x2)
    if not present:
        return {"n_auxiliary_signs":0,"energy_inside_sign":None,"peak_inside_sign":None}
    total=float(heat.sum())
    peak=np.unravel_index(int(np.argmax(heat)),heat.shape)
    return {"n_auxiliary_signs":present,
            "energy_inside_sign":float(heat[mask].sum()/total) if total>0 else None,
            "peak_inside_sign":bool(mask[peak]) if total>0 else None}
