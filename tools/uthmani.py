"""Uthmani (Tanzil) metni -> phonetiser'ın beklediği standart harekeli yazım."""
import re


def uth2msa(t: str) -> str:
    t = t.replace("ٱ", "ا")                                   # ٱ (vasl elifi) -> ا
    t = re.sub("(?<=[ء-يً-ْ])آ", "ا", t)  # kelime içi آ = med elifi
    t = t.replace("ۡ", "ْ")                                   # ۡ -> ْ (sükun)
    t = re.sub("َ?ٰ", "َا", t)                     # hançerî elif -> fetha + elif
    t = t.replace("ۥ", "و").replace("ۦ", "ي")      # küçük vav / ye (sıla)
    t = re.sub("[ۖ-ۜ۞-ۤۧ-ۭٕٓٔ]", "", t)  # durak, med vb.
    t = re.sub("([ً-ِ])ّ", "ّ\\1", t)               # şedde harekeden önce
    return t
