import numpy as np
import cv2
import argparse, pathlib, re, sys, os, time

EPS = 0.1
PIXEL_EPS = 10
RATIO_LOWER = 2 / 3 - EPS
RATIO_HIGHER = 2 / 3 + EPS
SEARCH_RANGE = 10


def crop(img):
    while True:
        if np.all(img[:, 0] == img[0, 0]):
            img = img[:, 1:]
        elif np.all(img[:, -1] == img[-1, -1]):
            img = img[:, :-1]
        elif np.all(img[0, :] == img[0, 0]):
            img = img[1:, :]
        elif np.all(img[-1, :] == img[-1, -1]):
            img = img[:-1, :]
        else:
            break
    return img


def load_and_preproc(img_paths, opts):
    rv = []
    for i, img_path in enumerate(img_paths):
        print(f"{i+1}/{len(img_paths)}", end="\r", flush=True)
        img = cv2.imread(img_path)
        if opts["crop"]:
            img = crop(img)
        if opts["split"]:
            ratio = img.shape[1] / img.shape[0]
            if not (RATIO_HIGHER > ratio and ratio > RATIO_LOWER) and (
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
                    split_imgs = [crop(im) for im in split_imgs]
                rv = rv + split_imgs
            else:
                rv.append(img)
        else:
            rv.append(img)
    print()
    return rv


# based on: https://www.geeksforgeeks.org/image-registration-using-opencv-python/
def align(to_align, refim, match_threshold):
    to_align_color = to_align
    refim_color = refim

    # Convert to grayscale.
    to_align_grey = cv2.cvtColor(to_align_color, cv2.COLOR_BGR2GRAY)
    refim_grey = cv2.cvtColor(refim_color, cv2.COLOR_BGR2GRAY)
    ref_height, ref_width = refim_grey.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(to_align_grey, None)
    kp2, d2 = orb_detector.detectAndCompute(refim_grey, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = list(matcher.match(d1, d2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # print(len(matches))
    if len(matches) < match_threshold:
        return

    # Take the top 90 % matches forward.
    matches = matches[: int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(
        to_align_color, homography, (ref_width, ref_height)
    )

    return transformed_img


def comp(a):
    rv = []
    for e in re.split("-| ", a):
        rv.append(len(e))
        rv.append(e)
    return rv


def main(config):
    refpath = config.original
    aligpath = config.translation

    refs = [os.path.join(refpath, p) for p in sorted(os.listdir(refpath), key=comp)]
    toaligs = [
        os.path.join(aligpath, p) for p in sorted(os.listdir(aligpath), key=comp)
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

    print("Loading images!")

    ref_imgs = load_and_preproc(refs, preproc_opts_refs)
    alig_imgs = load_and_preproc(toaligs, preproc_opts_alig)
    print()

    print(
        f"""Length of original: {len(ref_imgs)}
Length of translation: {len(alig_imgs)}
Original configuration: {preproc_opts_refs}
Translation configuration: {preproc_opts_alig}"""
    )
    print()
    print("Starting alignment!")

    aidx = 0
    lasthit = 0
    aligned = [[im] for im in ref_imgs]

    for ridx in range(len(ref_imgs)):
        print(f"{ridx+1}/{len(ref_imgs)}", end="\r", flush=True)
        rimg = ref_imgs[ridx]

        for offset in range(SEARCH_RANGE):
            if aidx + offset >= len(alig_imgs) or (
                lasthit >= aidx and aidx + offset - 1 > lasthit
            ):
                break
            aimg = alig_imgs[aidx + offset]
            transformed_img = align(aimg, rimg, config.match_threshold)
            if transformed_img is not None:
                aligned[ridx].append(transformed_img)
                lasthit = aidx + offset
                for btidx in range(offset):
                    if ridx <= btidx or len(aligned[ridx - btidx - 1]) != 1:
                        break
                    aligned[ridx - btidx - 1].append(alig_imgs[aidx - btidx])
        aidx = lasthit + 1
    print("\n")

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
                        imgset[idx][: overlaid.shape[0], : overlaid.shape[1]],
                        1 - (idx / setlength),
                        0,
                    )
                except:
                    print(f"Failed to overlay image no.{i}")
            cv2.imwrite(os.path.join(config.out, f"{i:04}-zoverlaid.png"), overlaid)

    print("\n")


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
        "--match-threshold",
        default=1150,
        type=int,
        help="threshold score for matching pages (default 1150)",
    )
    parser.add_argument(
        "--overlay",
        default="n",
        help="output overlaid versions for debugging (y/n, default no)",
    )

    args = parser.parse_args(sys.argv[1:])
    start_time = time.time()
    main(args)
    print(f"Finished in {time.time() - start_time} seconds!")
