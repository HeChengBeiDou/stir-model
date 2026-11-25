import numpy as np
from scipy.ndimage import label,binary_fill_holes
def clean_mask(mask):
    """固定常规后处理：8连通最大分量+孔洞填充；无GT、无时序平滑。"""
    components, n = label(mask, np.ones((3, 3)))
    if n == 0:
        return mask
    counts = np.bincount(components.ravel())
    counts[0] = 0
    return binary_fill_holes(components == counts.argmax())
