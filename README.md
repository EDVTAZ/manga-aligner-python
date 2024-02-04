# Manga Aligner

Takes two different language versions of the same manga/comic. Transforms (offset, zoom, etc) one of the versions, so the pages from the two different versions line up perfectly. This allows reading one page in a foreign language then crosschecking your understanding by reading the same in translation and then moving on to the next one. Alignment ensures that the images don't jump around and pages stay in sync.

# Usage

```
usage: Manga Aligner [-h] --original ORIGINAL --translation TRANSLATION --out OUT [--crop-orig CROP_ORIG] [--crop-transl CROP_TRANSL] [--split-orig SPLIT_ORIG]
                     [--split-transl SPLIT_TRANSL] [--dir-orig DIR_ORIG] [--dir-transl DIR_TRANSL] [--match-threshold MATCH_THRESHOLD] [--overlay OVERLAY]

Align the pages of versions of manga in different languages, to allow easy crosschecking while reading in a foreign language.

optional arguments:
  -h, --help            show this help message and exit
  --original ORIGINAL   original version, the language you are primarily reading in; can be a directory containing images(, cbz, cbr, cbt TODO)
  --translation TRANSLATION
                        translated version, the language you will use to check your understanding of the original; can be a directory containing images
  --out OUT             directory for output of aligned images
  --crop-orig CROP_ORIG
                        try to crop uniform color edges off the images in the original version (y/n, default no)
  --crop-transl CROP_TRANSL
                        try to crop uniform color edges off the images in the translated version (y/n, default no)
  --split-orig SPLIT_ORIG
                        try to split the images in the middle when two page images are detected in the original version (y/n, default yes)
  --split-transl SPLIT_TRANSL
                        try to split the images in the middle when two page images are detected in the translated version (y/n, default yes)
  --dir-orig DIR_ORIG   reading direction in the original (rl: right-to-left / lr: left-to-right; default rl)
  --dir-transl DIR_TRANSL
                        reading direction in the translation (rl: right-to-left / lr: left-to-right; default rl)
  --match-threshold MATCH_THRESHOLD
                        threshold score for matching pages (default 1150)
  --overlay OVERLAY     output overlaid versions for debugging (y/n, default no)
```

# TODO

- handle other formats (, cbz, cbr, cbt, TODO)
- demonstration images/movie in README
- optimization (downscaling, better cropping, etc.)
- correctness (threshold, allowed transformations),
- testing
