"""Build trivial_name_tokens.tsv: the OPSIN dictionary tokens that name whole
molecules or trivial (natural-product, drug, common) parents.

OPSIN (the jar shipped with the pinned py2opsin) resolves trivial names such as
caffeine or 5-fluorouracil, so the grader parses names a second time with these
tokens removed from OPSIN's dictionary (see iupac.py). A name that only parses
with them is not a systematic IUPAC name.

Scope: the token lists below whose entries are parent compounds or whole
molecules. Every token in them is blocked unless it is in ALLOWED, the names
that the IUPAC 2013 recommendations retain for general nomenclature
(Chapters P-2 to P-7, https://iupac.qmul.ac.uk/BlueBook/). Names that belong to
natural-product and biochemical nomenclature (Chapter P-10: carbohydrates,
amino acids, nucleosides, steroids, alkaloids, terpenes) and names the
recommendations no longer accept are blocked. Substituent prefixes, fusion
prefixes, and systematic morphemes (alkane stems, Hantzsch-Widman parts,
suffixes, inorganic oxoacids) are out of scope.

The output lists every in-scope token with its decision, so the list can be
reviewed and regenerated:

    uv run --no-project --with py2opsin==1.2.0 python build_trivial_name_tokens.py
"""
import csv
import importlib.util
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

OPSIN_JAR = (Path(importlib.util.find_spec("py2opsin").origin).parent
             / "opsin-cli-2.9.0-jar-with-dependencies.jar")
RESOURCES = "uk/ac/cam/ch/wwmm/opsin/resources/"
OUTPUT = Path(__file__).parent / "trivial_name_tokens.tsv"


def _all(attrs: dict) -> bool:
    return True


# file -> predicate on a <tokenList tagname="group"> element's attributes.
IN_SCOPE = {
    "arylGroups.xml": _all,
    "simpleGroups.xml": lambda a: a.get("subType") in ("simpleGroup", "biochemical") and a.get("symbol") in ("G", "Ø"),
    "simpleCyclicGroups.xml": _all,
    "naturalProducts.xml": _all,
    "aminoAcids.xml": _all,
    "carbohydrates.xml": lambda a: a.get("subType") not in ("systematicCarbohydrateStemAldose",)
                                   and a.get("type") == "carbohydrate",
    "carboxylicAcids.xml": _all,
    "cyclicUnsaturableHydrocarbon.xml": _all,
    "groupStemsAllowingInlineSuffixes.xml": _all,
    "groupStemsAllowingAllSuffixes.xml": _all,
    "simpleSubstituents.xml": lambda a: a.get("subType") == "biochemical",
}

P22_1_3 = "P-22.1.3 retained parent hydrocarbons"
T2_2 = "P-22.2.1 Table 2.2 retained mancude heteromonocycles"
T2_3 = "P-22.2.1 Table 2.3 retained saturated heteromonocycles"
T2_6 = "P-23.7 Table 2.6 retained von Baeyer names"
T2_7 = "P-25.1.1 Table 2.7 retained fused hydrocarbons"
P25_1_2 = "P-25.1.2 systematically named fused hydrocarbons"
T2_8 = "P-25.2.1 Table 2.8 retained fused heterocycles"
T2_9 = "P-25.2.1 Table 2.9 P/As analogues of retained fused heterocycles"
P25_2_2_2 = "P-25.2.2.2 heteranthrenes"
P25_2_2_3 = "P-25.2.2.3 pheno...ine names"
P25_2_2_4 = "P-25.2.2.4 benzo heterocycles (isobenzofuran)"
P25_3 = "P-25.3 fusion name"
T3_1 = "P-31.2.3.3.1 Table 3.1 retained partially saturated polycycles"
P31_1_2_1 = "P-31.1.2.1 retained acyclic parent hydrides"
P31_1_3_4 = "P-31.1.3.4 retained ring-chain hydrocarbons"
P21_1_1_2 = "P-21.1.1.2 retained mononuclear hydrides"
P34 = "P-34.1 retained functional parents"
P61_3_4 = "P-61.3.4 retained haloforms"
P62_2_1_1 = "P-62.2.1.1 retained amine names"
P63_1_1 = "P-63.1.1 retained hydroxy compounds"
P63_2_3 = "P-63.2.3 retained ether name"
P64_2_1 = "P-64.2.1 retained ketone names"
P65_1_1_1 = "P-65.1.1.1 retained acids (PIN)"
P65_1_1_2 = "P-65.1.1.2 retained acids (general nomenclature)"
P65_2 = "P-65.2 retained carbon acids"
P66 = "P-66 retained amide/amidine names"
P68_3 = "P-68.3 retained nitrogen parent names"
P72_P73 = "P-72/P-73 retained ion names"
P73_1_1 = "P-73.1.1 onium/ylium parent cations"
CLASS_TERM = "systematic suffix or class term, not a compound name"

# Per resource file: exact OPSIN token texts (alternatives joined by '|') that
# are allowed, with the IUPAC 2013 section that retains the name.
ALLOWED = {
    "arylGroups.xml": {
        "benzen": "P-22.1.1 benzene",
        "toluen": P22_1_3, "xylen": P22_1_3, "mesitylen": P22_1_3,
        "furan": T2_2, "imidazol": T2_2, "isoxazol|isooxazol": T2_2, "isothiazol": T2_2,
        "isoselenazol": T2_2, "isotellurazol": T2_2, "pyran": T2_2, "thiopyran": T2_2,
        "selenopyran": T2_2, "telluropyran": T2_2, "pyrazin": T2_2, "pyrazol": T2_2,
        "pyridazin": T2_2, "pyridin": T2_2, "pyrimidin": T2_2, "pyrrol": T2_2,
        "selenophen": T2_2, "tellurophen": T2_2, "thiophen": T2_2,
        "isoxazolidin|isooxazolidin": T2_3, "isothiazolidin": T2_3, "isoselenazolidin": T2_3,
        "isotellurazolidin": T2_3, "pyrrolidin": T2_3, "morpholin": T2_3,
        "thiamorpholin|thiomorpholin": T2_3, "selenomorpholin": T2_3, "telluromorpholin": T2_3,
        "pyrazolidin": T2_3, "imidazolidin": T2_3, "piperidin": T2_3, "piperazin": T2_3,
        "quinuclidin": T2_6,
        "ovalen": T2_7, "pyranthren": T2_7, "coronen": T2_7, "rubicen": T2_7, "perylen": T2_7,
        "picen": T2_7, "pleiaden": T2_7, "chrysen": T2_7, "pyren": T2_7, "fluoranthen": T2_7,
        "anthracen": T2_7, "phenanthren": T2_7, "phenalen": T2_7, "fluoren": T2_7,
        "s-indacen": T2_7, "as-indacen": T2_7, "azulen": T2_7, "naphthalen|naphthalin": T2_7,
        "inden": T2_7,
        "acenaphthylen": P25_1_2, "aceanthrylen": P25_1_2, "acephenanthrylen": P25_1_2,
        "biphenylen": P25_1_2, "triphenylen": P25_1_2, "trinaphthylen": P25_1_2,
        "cyclopenta[a]phenanthren": P25_3,
        "phenazin": T2_8, "phenanthrolin": T2_8, "perimidin": T2_8, "acridin": T2_8,
        "phenanthridin": T2_8, "carbazol": T2_8, "pteridin": T2_8, "cinnolin": T2_8,
        "quinazolin": T2_8, "quinoxalin": T2_8, "naphthyridin": T2_8, "phthalazin": T2_8,
        "quinolin": T2_8, "isoquinolin": T2_8, "quinolizin": T2_8, "purin": T2_8,
        "indazol": T2_8, "indol": T2_8, "isoindol": T2_8, "indolizin": T2_8, "pyrrolizin": T2_8,
        "xanthen": T2_8, "thioxanthen": T2_8, "selenoxanthen": T2_8, "telluroxanthen": T2_8,
        "chromen": T2_8, "thiochromen": T2_8, "selenochromen": T2_8, "tellurochromen": T2_8,
        "isochromen": T2_8, "isothiochromen": T2_8, "isoselenochromen": T2_8,
        "isotellurochromen": T2_8,
        "chromenylium": T2_8, "thiochromenylium": T2_8, "selenochromenylium": T2_8,
        "tellurochromenylium": T2_8, "isochromenylium": T2_8, "isothiochromenylium": T2_8,
        "isoselenochromenylium": T2_8, "isotellurochromenylium": T2_8,
        "acridarsin": T2_9, "acridophosphin": T2_9, "arsindol": T2_9, "phosphindol": T2_9,
        "arsindolizin": T2_9, "phosphindolizin": T2_9, "isoarsindol": T2_9,
        "isophosphindol": T2_9, "isoarsinolin": T2_9, "isophosphinolin": T2_9,
        "arsanthridin": T2_9, "phosphanthridin": T2_9, "arsinolin": T2_9, "phosphinolin": T2_9,
        "arsinolizin": T2_9, "phosphinolizin": T2_9,
        "oxanthren": P25_2_2_2, "thianthren": P25_2_2_2, "selenanthren": P25_2_2_2,
        "telluranthren": P25_2_2_2, "phosphanthren": P25_2_2_2, "arsanthren": P25_2_2_2,
        "boranthren": P25_2_2_2, "silanthren": P25_2_2_2,
        "phenoxazin": P25_2_2_3, "phenothiazin": P25_2_2_3, "phenoselenazin": P25_2_2_3,
        "phenotellurazin": P25_2_2_3, "phenophosphazinin|phenophosphazin": P25_2_2_3,
        "phenarsazin": P25_2_2_3, "phenarsazinin": P25_2_2_3, "phenoarsazin": P25_2_2_3,
        "phenazasilin": P25_2_2_3, "phenoxathiin": P25_2_2_3, "phenoxaselenin": P25_2_2_3,
        "phenoxatellurin": P25_2_2_3, "phenoxasilin": P25_2_2_3,
        "phenoxaphosphinin|phenoxaphosphin": P25_2_2_3, "phenoxarsinin|phenoxarsin": P25_2_2_3,
        "phenoxastibinin|phenoxantimonin": P25_2_2_3, "phenothiarsinin|phenothiarsin": P25_2_2_3,
        "isobenzofuran": P25_2_2_4, "isobenzothiofuran|isobenzothiophen": P25_2_2_4,
        "indan": T3_1, "indolin": T3_1, "isoindolin": T3_1, "chroman": T3_1,
        "thiochroman": T3_1, "selenochroman": T3_1, "tellurochroman": T3_1, "isochroman": T3_1,
        "isothiochroman": T3_1, "isoselenochroman": T3_1, "isotellurochroman": T3_1,
        "styren": P31_1_3_4, "stilben": P31_1_3_4, "fulven": P31_1_3_4,
        "anilin": P34, "anisol": P63_2_3,
        "phenoxid": P72_P73 + " (phenoxide)", "phenoxylium": P72_P73 + " (phenoxylium)",
        "pyrylium": P72_P73 + " (pyrylium)",
        "cresol": P63_1_1, "resorcinol": P63_1_1, "hydroquinon": P63_1_1,
        "pyrocatechol": P63_1_1,
        "chalcon": P64_2_1, "benzoquinon": P64_2_1 + " (1,4-benzoquinone)",
        "naphthoquinon": P64_2_1, "anthraquinon": P64_2_1,
        },
    "simpleGroups.xml": {
        "carbodiimide|carbodiimid": P31_1_2_1,
        "chloroform": P61_3_4, "bromoform": P61_3_4, "iodoform": P61_3_4,
        "cyanamide": P66 + " (cyanamide)",
        "cyanic|cyanicacid|cyanic acid": P65_2,
        "hydrocyanicacid|hydrocyanic acid": "P-66.5 hydrocyanic acid",
        "formamide|formamid": P66 + " (formamide)", "methanamide|methanamid": "systematic",
        "glyoxal": P34, "oxamide|oxamid": P34, "urea": P34,
        "semicarbazide|semicarbazid": P68_3 + " (semicarbazide)",
        "ammonia": P21_1_1_2,
        "pentaerythritol": P63_1_1, "pinacol": P63_1_1,
        "peracetic|peraceticacid|peracetic acid": P65_1_1_2, "peracetate|peracetat": P65_1_1_2,
        "performic|performicacid|performic acid": P65_1_1_2, "performate": P65_1_1_2,
        "nitramide|nitramid": P68_3 + " (nitramide)",
        "uronium": P73_1_1 + " (uronium)",
        "methoxide|methoxid": P72_P73, "ethoxide|ethoxid": P72_P73,
        "propoxide|n-propoxide|propoxid|n-propoxid": P72_P73,
        "butoxide|n-butoxide|butoxid|n-butoxid": P72_P73,
        "methoxylium": P72_P73, "ethoxylium": P72_P73, "propoxylium": P72_P73,
        "butoxylium": P72_P73, "aminoxide|aminoxid": P72_P73, "aminoxylium": P72_P73,
        "aminylium": P72_P73, "nitrenium": P72_P73,
        "acetylide|acetylid": CLASS_TERM, "amine|amin": CLASS_TERM, "aminium": CLASS_TERM,
        "aminide|aminid": CLASS_TERM, "carboxamide|carboxamid": CLASS_TERM,
        "carboxylate|carboxylat": CLASS_TERM, "carboxylic|carboxylicacid|carboxylic acid": CLASS_TERM,
        "diazonium": CLASS_TERM, "nitrone": CLASS_TERM,
        "sulfoxonium": CLASS_TERM, "sulfurane|sulfuran": CLASS_TERM, "selenurane|selenuran": CLASS_TERM,
        "persulfurane|persulfuran": CLASS_TERM, "perselenurane|perselenuran": CLASS_TERM,
        "sulfoximide|sulfoximine|sulfoximid|sulfoximin": CLASS_TERM,
        },
    "simpleCyclicGroups.xml": {
        "phenol": P34, "phenolate|phenolat": P72_P73 + " (phenolate)",
        "thymol": P63_1_1, "carvacrol": P63_1_1, "picric|picricacid|picric acid": P63_1_1,
        "perbenzoic|perbenzoicacid|perbenzoic acid": P65_1_1_2, "perbenzoate|perbenzoat": P65_1_1_2,
        "benzo-1,4-quinone|benzo-1,4-quinon": P64_2_1 + " (1,4-benzoquinone)",
        "naphtho-1,2-quinone|naphtho-1,2-quinon": P64_2_1, "naphtho-1,4-quinone|naphtho-1,4-quinon": P64_2_1,
        "anthra-1,2-quinone|anthra-1,2-quinon": P64_2_1, "anthra-1,4-quinone|anthra-1,4-quinon": P64_2_1,
        "anthra-9,10-quinone|anthra-9,10-quinon": P64_2_1,
        },
    "carboxylicAcids.xml": {
        "form": P65_1_1_1, "acet": P65_1_1_1, "oxal": P65_1_1_1, "benz": P65_1_1_1,
        "oxam": P65_1_1_1,
        "phthal": P65_1_1_2, "isophthal": P65_1_1_2, "terephthal": P65_1_1_2,
        "propion|propi": P65_1_1_2, "butyr": P65_1_1_2, "malon": P65_1_1_2, "succin": P65_1_1_2,
        "glutar": P65_1_1_2, "adip": P65_1_1_2, "acryl": P65_1_1_2, "methacryl": P65_1_1_2,
        "male": P65_1_1_2, "fumar": P65_1_1_2, "cinnam": P65_1_1_2, "nicotin": P65_1_1_2,
        "isonicotin": P65_1_1_2, "ole": P65_1_1_2, "palmit": P65_1_1_2, "stear": P65_1_1_2,
        "citr": P65_1_1_2, "glycer": P65_1_1_2, "pyruv": P65_1_1_2,
        "lact|dl-lact": P65_1_1_2, "l-lact|l(+)-lact|l-(+)-lact": P65_1_1_2,
        "d-lact|d(-)-lact|d-(-)-lact": P65_1_1_2,
        "tartar": P65_1_1_2, "dl-tartar": P65_1_1_2, "l-tartar": P65_1_1_2,
        "l(+)-tartar": P65_1_1_2, "l-(+)-tartar": P65_1_1_2, "d-tartar": P65_1_1_2,
        "d(-)-tartar": P65_1_1_2, "d-(-)-tartar": P65_1_1_2, "mesotartar": P65_1_1_2,
        "edet": P65_1_1_2 + " (ethylenediaminetetraacetic acid)",
        },
    "cyclicUnsaturableHydrocarbon.xml": {
        "adamant": T2_6, "cub": T2_6,
        },
    "groupStemsAllowingInlineSuffixes.xml": {
        "allen": P31_1_2_1, "formazan": P31_1_2_1, "keten": P64_2_1, "isopren": P31_1_2_1,
        "aceton": P64_2_1,
        "ammonium": P73_1_1, "phosphonium": P73_1_1, "arsonium": P73_1_1, "stibonium": P73_1_1,
        "bismuthonium": P73_1_1, "oxonium": P73_1_1, "sulfonium": P73_1_1, "selenonium": P73_1_1,
        "telluronium": P73_1_1, "fluoronium": P73_1_1, "chloronium": P73_1_1, "bromonium": P73_1_1,
        "iodonium": P73_1_1, "silylium": P73_1_1, "germylium": P73_1_1, "stannylium": P73_1_1,
        "plumbylium": P73_1_1,
        "phosphine": P21_1_1_2, "arsine": P21_1_1_2, "stibin": P21_1_1_2, "bismuthin": P21_1_1_2,
        "glycerol|glycerin": P63_1_1, "guanidin": P34,
        },
    "groupStemsAllowingAllSuffixes.xml": {
        "hydrazin": P68_3 + " (hydrazine)", "hydroxylamin": P34, "acetylen": P31_1_2_1,
        },
}

FIELDS = ["file", "token", "list", "decision", "reason", "smiles"]


def main() -> None:
    rows = []
    seen_allowed = set()
    with zipfile.ZipFile(OPSIN_JAR) as jar:
        for file, in_scope in IN_SCOPE.items():
            root = ET.fromstring(jar.read(RESOURCES + file))
            out_of_scope = set()
            in_scope_rows = []
            for token_list in root.iter("tokenList"):
                attrs = token_list.attrib
                scoped = attrs.get("tagname") == "group" and in_scope(attrs)
                label = "/".join(attrs.get(k, "") for k in ("type", "subType", "symbol"))
                for token in token_list.iter("token"):
                    text = token.text
                    if not text:
                        continue
                    if not scoped:
                        out_of_scope.add(text)
                        continue
                    if text in ALLOWED.get(file, {}):
                        decision, reason = "allowed", ALLOWED[file][text]
                        seen_allowed.add((file, text))
                    else:
                        decision, reason = "blocked", "not retained for general nomenclature"
                    in_scope_rows.append({"file": file, "token": text, "list": label,
                                          "decision": decision, "reason": reason,
                                          "smiles": token.get("value", "")})
            # iupac.py removes blocked tokens from a file by their text, so a blocked
            # text must not also occur in an out-of-scope list of the same file.
            clashes = {r["token"] for r in in_scope_rows if r["decision"] == "blocked"} & out_of_scope
            if clashes:
                raise SystemExit(f"{file}: blocked tokens also out of scope: {sorted(clashes)}")
            rows.extend(in_scope_rows)
    unused = {(f, t) for f, tokens in ALLOWED.items() for t in tokens} - seen_allowed
    if unused:
        raise SystemExit(f"ALLOWED entries not found in an in-scope list: {sorted(unused)}")
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    blocked = sum(r["decision"] == "blocked" for r in rows)
    print(f"{len(rows)} in-scope tokens: {blocked} blocked, {len(rows) - blocked} allowed -> {OUTPUT.name}")


if __name__ == "__main__":
    main()
