(TeX-add-style-hook
 "hwk7"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "10pt" "a4paper")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("geometry" "margin=1in")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "inputenc"
    "amsmath"
    "amsfonts"
    "amssymb"
    "graphics"
    "geometry"
    "amsthm")
   (LaTeX-add-environments
    "problem")))

