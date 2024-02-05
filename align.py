import numpy as np
import cv2
import argparse, pathlib, re, sys, os, time, pprint

EPS = 0.1
PIXEL_EPS = 10
RATIO_LOWER = 2 / 3 - EPS
RATIO_HIGHER = 2 / 3 + EPS
SEARCH_RANGE = 10


def crop(img):
    changed = True
    original = img
    try:
        while changed:
            changed = False
            if np.all(img[:, 0] == img[0, 0]):
                img = img[:, 1:]
                changed = True
            if np.all(img[:, -1] == img[-1, -1]):
                img = img[:, :-1]
                changed = True
            if np.all(img[0, :] == img[0, 0]):
                img = img[1:, :]
                changed = True
            if np.all(img[-1, :] == img[-1, -1]):
                img = img[:-1, :]
                changed = True
    except:
        return original

    return img


def double_page(img):
    ratio = img.shape[1] / img.shape[0]
    return not (RATIO_HIGHER > ratio and ratio > RATIO_LOWER)


def downscale(img, resize):
    if resize > 0:
        target_size = resize
        if double_page(img):
            target_size = 2 * target_size
        width = int(np.sqrt(target_size * (img.shape[1] / img.shape[0])))
        height = int(target_size / width)
        return cv2.resize(img, (width, height))


def load_and_preproc(img_paths, resize, opts):
    rv = []
    for i, img_path in enumerate(img_paths):
        print(f"{i+1}/{len(img_paths)}", end="\r", flush=True)
        img = cv2.imread(img_path)
        if opts["crop"]:
            img = crop(img)
        if opts["split"]:
            if double_page(img) and (
                np.all(img[:, int(img.shape[1] / 2)] == img[0, int(img.shape[1] / 2)])
                or np.all(
                    img[:, int(img.shape[1] / 2) - PIXEL_EPS]
                    == img[0, int(img.shape[1] / 2) - PIXEL_EPS]
                )
                or np.all(
                    img[:, int(img.shape[1] / 2) + PIXEL_EPS]
                    == img[0, int(img.shape[1] / 2) + PIXEL_EPS]
                )
            ):
                split_imgs = [
                    img[:, int(img.shape[1] / 2) :],
                    img[:, : int(img.shape[1] / 2)],
                ]
                if opts["dir_rl"] == False:
                    split_imgs = split_imgs[::-1]
                if opts["crop"]:
                    split_imgs = [downscale(crop(im), resize) for im in split_imgs]
                rv = rv + split_imgs
            else:
                rv.append(downscale(img, resize))
        else:
            rv.append(downscale(img, resize))
    print()
    return rv


def accept_homography(inl, ref_dim, alig_dim, homography):
    if inl <= 10:
        return False
    alig_height, alig_width = alig_dim
    contour = np.float32(
        [
            [0, 0],
            [0, alig_height - 1],
            [alig_width - 1, alig_height - 1],
            [alig_width - 1, 0],
        ]
    ).reshape(-1, 1, 2)
    contour = cv2.perspectiveTransform(contour, homography)
    if not cv2.isContourConvex(contour):
        return False
    ref_area = ref_dim[0] * ref_dim[1]
    alig_area = cv2.contourArea(contour)
    ref_alig_ratio = ref_area / alig_area
    if ref_alig_ratio > 3 or ref_alig_ratio < 1 / 3:
        return False
    return True


def get_homography(to_align_grey, refim_grey):
    orb_detector = cv2.ORB_create(5000)

    kp_alig, d_alig = orb_detector.detectAndCompute(to_align_grey, None)
    kp_ref, d_ref = orb_detector.detectAndCompute(refim_grey, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    try:
        matches = list(matcher.knnMatch(d_alig, d_ref, k=2))
    except:
        return [None, None]

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x[0].distance)

    # Take the top 90 % matches forward.
    matches = matches[: int(len(matches) * 0.9)]

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    matches = good

    if len(matches) < 4:
        return [None, None]

    # Define empty matrices of shape no_of_matches * 2.
    mp_alig = np.zeros((len(matches), 2))
    mp_ref = np.zeros((len(matches), 2))

    for i in range(len(matches)):
        mp_alig[i, :] = kp_alig[matches[i].queryIdx].pt
        mp_ref[i, :] = kp_ref[matches[i].trainIdx].pt

    return cv2.findHomography(mp_alig, mp_ref, cv2.RANSAC)


def align(to_align, refim):
    to_align_color = to_align
    refim_color = refim

    # Convert to grayscale.
    to_align_grey = cv2.cvtColor(to_align_color, cv2.COLOR_BGR2GRAY)
    refim_grey = cv2.cvtColor(refim_color, cv2.COLOR_BGR2GRAY)
    ref_height, ref_width = refim_grey.shape

    homography, mask = get_homography(to_align_grey, refim_grey)
    if homography is None:
        return None

    mask_cnt = np.unique(mask, return_counts=True)
    if len(mask_cnt[1]) > 1:
        inl = mask_cnt[1][1]
    else:
        inl = 0
    if not accept_homography(inl, refim_grey.shape, to_align_grey.shape, homography):
        return None

    transformed_img = cv2.warpPerspective(
        to_align_color, homography, (ref_width, ref_height)
    )
    return transformed_img


def comp_filename(a):
    rv = []
    for e in re.split("-| ", a):
        rv.append(len(e))
        rv.append(e)
    return rv


def batched_align(ref_imgs, alig_imgs):
    print("Starting alignment!")

    aidx = 0
    lasthit = -1
    aligned = [[im] for im in ref_imgs]

    for ridx in range(len(ref_imgs)):
        print(f"{ridx+1}/{len(ref_imgs)}", end="\r", flush=True)
        rimg = ref_imgs[ridx]

        for offset in range(SEARCH_RANGE):
            if (
                aidx + offset >= len(alig_imgs)
                or (lasthit >= aidx and aidx + offset > lasthit + 1)
                or (lasthit >= aidx and not double_page(rimg))
            ):
                break
            aimg = alig_imgs[aidx + offset]
            transformed_img = align(aimg, rimg)
            if transformed_img is not None:
                aligned[ridx].append(transformed_img)
                lasthit = aidx + offset
                for btidx in range(offset):
                    if (
                        aidx < btidx
                        or ridx <= btidx
                        or len(aligned[ridx - btidx - 1]) != 1
                    ):
                        break
                    print(
                        f"WARNING: backtrace auto-fill for image no.{ridx - btidx - 1}"
                    )
                    aligned[ridx - btidx - 1].append(
                        alig_imgs[aidx + offset - btidx - 1]
                    )
        aidx = lasthit + 1
    print("\n")
    return aligned


def save_imgs(aligned, config):
    print("Saving images!")
    for i, imgset in enumerate(aligned):
        print(f"{i+1}/{len(aligned)}", end="\r", flush=True)
        timg = None
        baseimg = imgset[0]

        if len(imgset) > 1:
            timg = imgset[1]
        else:
            print(f"WARNING: missed matching translation for output image no.{i}")
        if len(imgset) > 2:
            setlength = len(imgset) - 1
            for idx in range(setlength)[1:]:
                timg = cv2.addWeighted(
                    timg,
                    idx / setlength,
                    imgset[idx + 1][: timg.shape[0], : timg.shape[1]],
                    1 - (idx / setlength),
                    0,
                )

        cv2.imwrite(os.path.join(config.out, f"{i:04}-orig.png"), baseimg)
        if timg is not None:
            cv2.imwrite(os.path.join(config.out, f"{i:04}-transl.png"), timg)

        if config.overlay:
            overlaid = baseimg
            setlength = len(imgset)
            for idx in range(setlength)[1:]:
                try:
                    overlaid = cv2.addWeighted(
                        overlaid,
                        idx / setlength,
                        cv2.resize(imgset[idx], overlaid.shape[:2][::-1]),
                        1 - (idx / setlength),
                        0,
                    )
                except:
                    print(f"Failed to overlay image no.{i}")
            cv2.imwrite(os.path.join(config.out, f"{i:04}-zoverlaid.png"), overlaid)

    print("\n")


def main(config):
    refpath = config.original
    aligpath = config.translation

    refs = [
        os.path.join(refpath, p) for p in sorted(os.listdir(refpath), key=comp_filename)
    ]
    toaligs = [
        os.path.join(aligpath, p)
        for p in sorted(os.listdir(aligpath), key=comp_filename)
    ]

    preproc_opts_refs = {
        "crop": config.crop_orig == "y",
        "split": config.split_orig == "y",
        "dir_rl": config.dir_orig == "rl",
    }
    preproc_opts_alig = {
        "crop": config.crop_transl == "y",
        "split": config.split_transl == "y",
        "dir_rl": config.dir_transl == "rl",
    }

    pprint.pprint(vars(config))
    print()

    print("Loading images!")
    ref_imgs = load_and_preproc(refs, config.downscale, preproc_opts_refs)
    alig_imgs = load_and_preproc(toaligs, config.downscale, preproc_opts_alig)
    print()

    aligned = batched_align(ref_imgs, alig_imgs)

    save_imgs(aligned, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Manga Aligner",
        description="Align the pages of versions of manga in different languages, to allow easy crosschecking while reading in a foreign language.",
    )
    parser.add_argument(
        "--original",
        type=pathlib.Path,
        required=True,
        help="original version, the language you are primarily reading in; can be a directory containing images(, cbz, cbr, cbt TODO)",
    )
    parser.add_argument(
        "--translation",
        type=pathlib.Path,
        required=True,
        help="translated version, the language you will use to check your understanding of the original; can be a directory containing images",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="directory for output of aligned images",
    )
    parser.add_argument(
        "--crop-orig",
        default="n",
        help="try to crop uniform color edges off the images in the original version (y/n, default no)",
    )
    parser.add_argument(
        "--crop-transl",
        default="n",
        help="try to crop uniform color edges off the images in the translated version (y/n, default no)",
    )
    parser.add_argument(
        "--split-orig",
        default="y",
        help="try to split the images in the middle when two page images are detected in the original version (y/n, default yes)",
    )
    parser.add_argument(
        "--split-transl",
        default="y",
        help="try to split the images in the middle when two page images are detected in the translated version (y/n, default yes)",
    )
    parser.add_argument(
        "--dir-orig",
        default="rl",
        help="reading direction in the original (rl: right-to-left / lr: left-to-right; default rl)",
    )
    parser.add_argument(
        "--dir-transl",
        default="rl",
        help="reading direction in the translation (rl: right-to-left / lr: left-to-right; default rl)",
    )
    parser.add_argument(
        "--overlay",
        default="n",
        help="output overlaid versions for debugging (y/n, default no)",
    )
    parser.add_argument(
        "--downscale",
        default=2000000,
        type=int,
        help="reduce resolution to this many pixels on single pages (default: 2000000, -1 for no resize)",
    )

    args = parser.parse_args(sys.argv[1:])
    start_time = time.time()
    main(args)
    print(f"Finished in {time.time() - start_time} seconds!")
