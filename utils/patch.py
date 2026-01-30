# utils/patch.py

def extract_patches(img, mask, patch_size, stride):
    h, w = img.shape
    ps = patch_size

    for y in range(0, h - ps + 1, stride):
        for x in range(0, w - ps + 1, stride):
            yield (
                img[y:y+ps, x:x+ps],
                mask[y:y+ps, x:x+ps]
            )
