"""Prepare a bounded, LOCAL-ONLY IPUMS CPU pilot sample.

This is a public-use-data research helper, not a DP release. The NPZ contains
encoded real records and MUST NOT be published. Only seven modeling variables
are retained. A fixed public age transformation and documented categorical
domains avoid fitting preprocessing statistics to the sampled records.

OCC and IND domains come from the public Census 2024 PUMS dictionary, plus
IPUMS not-applicable code zero; they are NOT inferred from the sampled records.
Other categorical domains come from the supplied public SAS format metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from datetime import datetime

import numpy as np


FEATURES = ["AGE", "SEX", "RACE", "HISPAN", "EDUCD", "OCC", "IND"]


OCC_2024_CODES = [
    0, 10, 20, 40, 51, 52, 60, 101, 102, 110, 120, 135, 136, 137, 140, 150,
    160, 205, 220, 230, 300, 310, 335, 340, 350, 360, 410, 420, 425, 440, 500, 510,
    520, 530, 540, 565, 600, 630, 640, 650, 700, 705, 710, 725, 726, 735, 750, 800,
    810, 820, 830, 845, 850, 860, 900, 910, 930, 940, 960, 1005, 1006, 1007, 1010, 1021,
    1022, 1031, 1032, 1050, 1065, 1105, 1106, 1108, 1200, 1220, 1240, 1305, 1306, 1310, 1320, 1340,
    1350, 1360, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1520, 1530, 1541, 1545, 1551, 1555, 1560,
    1600, 1610, 1640, 1650, 1700, 1710, 1720, 1745, 1750, 1760, 1800, 1821, 1822, 1825, 1840, 1860,
    1900, 1910, 1920, 1935, 1970, 1980, 2001, 2002, 2003, 2004, 2005, 2006, 2011, 2012, 2013, 2014,
    2015, 2016, 2025, 2040, 2050, 2060, 2100, 2105, 2145, 2170, 2180, 2205, 2300, 2310, 2320, 2330,
    2350, 2360, 2400, 2435, 2440, 2545, 2555, 2600, 2631, 2632, 2633, 2634, 2635, 2636, 2640, 2700,
    2710, 2721, 2722, 2723, 2740, 2751, 2752, 2755, 2770, 2805, 2810, 2825, 2830, 2840, 2850, 2861,
    2862, 2865, 2905, 2910, 2920, 3000, 3010, 3030, 3040, 3050, 3090, 3100, 3110, 3120, 3140, 3150,
    3160, 3200, 3210, 3220, 3230, 3245, 3250, 3255, 3256, 3258, 3261, 3270, 3300, 3310, 3321, 3322,
    3323, 3324, 3330, 3401, 3402, 3421, 3422, 3423, 3424, 3430, 3500, 3515, 3520, 3545, 3550, 3601,
    3602, 3603, 3605, 3610, 3620, 3630, 3640, 3645, 3646, 3647, 3648, 3649, 3655, 3700, 3710, 3720,
    3725, 3740, 3750, 3801, 3802, 3820, 3840, 3870, 3900, 3910, 3930, 3940, 3945, 3946, 3960, 4000,
    4010, 4020, 4030, 4040, 4055, 4110, 4120, 4130, 4140, 4150, 4160, 4200, 4210, 4220, 4230, 4240,
    4251, 4252, 4255, 4330, 4340, 4350, 4400, 4420, 4435, 4461, 4465, 4500, 4510, 4521, 4522, 4525,
    4530, 4540, 4600, 4621, 4622, 4640, 4655, 4700, 4710, 4720, 4740, 4750, 4760, 4800, 4810, 4820,
    4830, 4840, 4850, 4900, 4920, 4930, 4940, 4950, 4965, 5000, 5010, 5020, 5040, 5100, 5110, 5120,
    5140, 5150, 5160, 5165, 5220, 5230, 5240, 5250, 5260, 5300, 5310, 5320, 5330, 5340, 5350, 5360,
    5400, 5410, 5420, 5500, 5510, 5521, 5522, 5530, 5540, 5550, 5560, 5600, 5610, 5630, 5710, 5720,
    5730, 5740, 5810, 5820, 5840, 5850, 5860, 5900, 5910, 5920, 5940, 6005, 6010, 6040, 6050, 6115,
    6120, 6130, 6200, 6210, 6220, 6230, 6240, 6250, 6260, 6305, 6330, 6355, 6360, 6400, 6410, 6441,
    6442, 6460, 6515, 6520, 6530, 6540, 6600, 6660, 6700, 6710, 6720, 6730, 6740, 6765, 6800, 6825,
    6835, 6850, 6950, 7000, 7010, 7020, 7030, 7040, 7100, 7120, 7130, 7140, 7150, 7160, 7200, 7210,
    7220, 7240, 7260, 7300, 7315, 7320, 7330, 7340, 7350, 7360, 7410, 7420, 7430, 7510, 7540, 7560,
    7610, 7640, 7700, 7720, 7730, 7740, 7750, 7800, 7810, 7830, 7840, 7850, 7855, 7905, 7925, 7950,
    8000, 8025, 8030, 8040, 8100, 8130, 8140, 8225, 8250, 8255, 8256, 8300, 8310, 8320, 8335, 8350,
    8365, 8450, 8465, 8500, 8510, 8530, 8540, 8555, 8600, 8610, 8620, 8630, 8640, 8650, 8710, 8720,
    8730, 8740, 8750, 8760, 8800, 8810, 8830, 8850, 8910, 8920, 8930, 8940, 8950, 8990, 9005, 9030,
    9040, 9050, 9110, 9121, 9122, 9130, 9141, 9142, 9150, 9210, 9240, 9265, 9300, 9310, 9350, 9365,
    9410, 9415, 9430, 9510, 9570, 9600, 9610, 9620, 9630, 9640, 9645, 9650, 9720, 9760, 9800, 9810,
    9825, 9830, 9920,
]

IND_2024_CODES = [
    0, 170, 180, 190, 270, 280, 290, 370, 380, 390, 470, 490, 570, 580, 590, 670,
    680, 690, 770, 1070, 1080, 1090, 1170, 1180, 1190, 1270, 1280, 1290, 1370, 1390, 1470, 1480,
    1490, 1570, 1590, 1670, 1691, 1770, 1790, 1870, 1880, 1890, 1990, 2070, 2090, 2170, 2180, 2190,
    2270, 2280, 2290, 2370, 2380, 2390, 2470, 2480, 2490, 2570, 2590, 2670, 2680, 2690, 2770, 2780,
    2790, 2870, 2880, 2890, 2970, 2980, 2990, 3070, 3080, 3095, 3170, 3180, 3291, 3365, 3370, 3380,
    3390, 3470, 3490, 3570, 3580, 3590, 3670, 3680, 3690, 3770, 3780, 3790, 3875, 3895, 3960, 3970,
    3980, 3990, 4070, 4080, 4090, 4170, 4180, 4195, 4265, 4270, 4280, 4290, 4370, 4380, 4390, 4470,
    4480, 4490, 4560, 4570, 4580, 4585, 4590, 4670, 4681, 4691, 4771, 4796, 4871, 4881, 4891, 4971,
    4973, 4981, 4991, 5071, 5081, 5090, 5171, 5181, 5191, 5276, 5281, 5296, 5371, 5382, 5392, 5471,
    5481, 5491, 5571, 5581, 5680, 5791, 6070, 6080, 6090, 6170, 6180, 6190, 6270, 6280, 6290, 6370,
    6380, 6390, 6471, 6481, 6490, 6570, 6590, 6671, 6680, 6690, 6695, 6770, 6781, 6871, 6881, 6890,
    6970, 6991, 6992, 7071, 7072, 7080, 7181, 7190, 7270, 7280, 7290, 7370, 7380, 7390, 7460, 7470,
    7480, 7490, 7570, 7580, 7590, 7670, 7680, 7690, 7770, 7780, 7790, 7860, 7870, 7880, 7890, 7970,
    7980, 7990, 8070, 8080, 8090, 8170, 8180, 8191, 8192, 8270, 8290, 8370, 8380, 8390, 8470, 8561,
    8562, 8563, 8564, 8570, 8580, 8590, 8660, 8670, 8680, 8690, 8770, 8780, 8790, 8870, 8891, 8970,
    8980, 8990, 9070, 9080, 9090, 9160, 9170, 9180, 9190, 9290, 9370, 9380, 9390, 9470, 9480, 9490,
    9570, 9590, 9670, 9680, 9690, 9770, 9780, 9790, 9870, 9920,
]

PUBLIC_CODE_SOURCE = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.txt"
PUBLIC_CODE_SHA256 = "9a98282277aa7f2dafb12b0bdf38b258689d3a1a9c2abbd9b5e2c1ddde6f7b32"


def read_schema(sas_path: Path) -> tuple[dict[str, tuple[int, int]], str]:
    text = sas_path.read_text(encoding="utf-8")
    match = re.search(r"(?ims)^\s*INPUT\s*\n(.*?);", text)
    if match is None:
        raise ValueError("Could not find the SAS INPUT block; use the matching .sas file.")
    fields = {
        name.upper(): (int(start) - 1, int(end))
        for name, start, end in re.findall(
            r"(?m)^\s*([A-Za-z][A-Za-z0-9_]*)\s+(\d+)\s*-\s*(\d+)",
            match.group(1),
        )
    }
    missing = sorted(set(FEATURES + ["YEAR", "SAMPLE"]) - fields.keys())
    if missing:
        raise ValueError(f"SAS INPUT is missing required variables: {missing}")
    return fields, text


def public_domains(fields: dict[str, tuple[int, int]], sas_text: str) -> dict[str, list[int]]:
    domains = {}
    for name in FEATURES[1:]:
        if name in {"OCC", "IND"}:
            start, end = fields[name]
            if end - start != 4:
                raise ValueError(f"Expected a four-digit {name} field in this pilot schema.")
            domains[name] = list(OCC_2024_CODES if name == "OCC" else IND_2024_CODES)
            continue
        match = re.search(
            rf"(?ims)^\s*value\s+{name}_f\s*\n(.*?);", sas_text
        )
        if match is None:
            raise ValueError(f"Missing public SAS format for {name}.")
        values = sorted({int(code) for code in re.findall(r"(?m)^\s*(\d+)\s*=", match.group(1))})
        if not values:
            raise ValueError(f"No public category codes were found for {name}.")
        domains[name] = values
    return domains


def sample_fixed_width(dat_path: Path, fields: dict[str, tuple[int, int]], rows: int, seed: int) -> list[dict[str, int]]:
    """Random-access sample from an uncompressed rectangular fixed-width file.

    Memory use is O(rows), not O(the entire extract). Counts and sampling seed
    here are research metadata; this is not a private sampling service.
    """
    if dat_path.suffix.lower() != ".dat":
        raise ValueError("Use the uncompressed .dat file, not .dat.gz or a ZIP.")
    with dat_path.open("rb") as stream:
        first = stream.readline()
        width = len(first.rstrip(b"\r\n"))
        stride = len(first)
        if width != max(end for _, end in fields.values()) or stride <= width:
            raise ValueError("The data row width does not match this SAS schema.")
        byte_count = dat_path.stat().st_size
        if byte_count % stride != 0:
            raise ValueError("File size is not an integer number of fixed-width rows; refusing to guess.")
        n_records = byte_count // stride
        if n_records < rows:
            raise ValueError(f"Requested {rows} rows, but the file contains only {n_records}.")
        indexes = random.Random(seed).sample(range(n_records), rows)
        sampled = []
        for index in indexes:
            stream.seek(index * stride)
            line = stream.read(stride)
            if len(line.rstrip(b"\r\n")) != width:
                raise ValueError("Inconsistent record length; refusing to parse shifted columns.")
            record = {}
            for name in FEATURES + ["YEAR", "SAMPLE"]:
                start, end = fields[name]
                token = line[start:end].strip()
                if not token.isdigit():
                    raise ValueError(f"Unexpected nonnumeric token in {name}; inspect the extract privately.")
                record[name] = int(token)
            sampled.append(record)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dat", type=Path, required=True)
    parser.add_argument("--sas", type=Path, default=Path("usa_00002.sas"))
    parser.add_argument("--train-rows", type=int, default=2048)
    parser.add_argument("--test-rows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=None, help="A NEW output directory; existing directories are not overwritten.")
    args = parser.parse_args()
    if not 16 <= args.train_rows <= 5000 or not 16 <= args.test_rows <= 5000:
        parser.error("Use 16–5000 training rows and 16–5000 held-out rows for this bounded pilot.")
    fields, sas_text = read_schema(args.sas)
    domains = public_domains(fields, sas_text)
    rows = sample_fixed_width(args.dat, fields, args.train_rows + args.test_rows, args.seed)
    if any(row["YEAR"] != 2024 or row["SAMPLE"] != 202401 for row in rows):
        raise ValueError("This pilot expects the old 2024 ACS sample (YEAR=2024, SAMPLE=202401); review any other extract first.")
    encoders = {name: {code: i for i, code in enumerate(codes)} for name, codes in domains.items()}
    matrix = np.empty((len(rows), len(FEATURES)), dtype=np.float32)
    for i, row in enumerate(rows):
        if not 0 <= row["AGE"] <= 140:
            raise ValueError("Encountered an unknown/missing AGE code. Do not silently treat it as an age.")
        matrix[i, 0] = (min(row["AGE"], 100) - 50.0) / 50.0
        for j, name in enumerate(FEATURES[1:], start=1):
            if row[name] not in encoders[name]:
                raise ValueError(f"A {name} value is absent from the public metadata domain. Do not infer a replacement from the data.")
            matrix[i, j] = encoders[name][row[name]]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = args.out or Path("pipeline_runs/prototype2_cpu") / f"ipums_data_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        out / "pilot_data.npz",
        X_train=matrix[:args.train_rows],
        X_test=matrix[args.train_rows:],
        categories=np.asarray([len(domains[name]) for name in FEATURES[1:]], dtype=np.int64),
    )
    metadata = {
        "purpose": "LOCAL-ONLY public-use IPUMS exploratory pilot, NOT a DP release",
        "features": FEATURES,
        "train_rows": args.train_rows,
        "test_rows": args.test_rows,
        "sampling": "uniform random row sample without replacement, followed by a disjoint row split",
        "sampling_seed": args.seed,
        "privacy_unit_for_future_accounting": "one record, not one household",
        "age_transform": "clip AGE to [0,100], then (AGE-50)/50; reject missing AGE codes",
        "category_domains": domains,
        "public_code_source": PUBLIC_CODE_SOURCE,
        "public_code_document_sha256": PUBLIC_CODE_SHA256,
        "domain_provenance": "OCC/IND numeric codes from public 2024 PUMS OCCP/INDP lists plus IPUMS zero; no categories learned from microdata. Other categories from public SAS format blocks.",
        "domain_warning": "Other SAS categorical formats may include codes from other years. No semantics or structural relationship constraints are imposed. IPUMS zero inclusion follows its public not-applicable convention; not an explicitly located 2024 Census-to-IPUMS crosswalk.",
        "omissions": "No income field available. PUMA omitted because state is unavailable. No identifiers or survey weights are modeled.",
        "weighting": "unweighted record sample, not a population-weighted estimate",
        "schema_sha256": hashlib.sha256(sas_text.encode()).hexdigest(),
        "do_not_publish": "pilot_data.npz contains encoded REAL records. Do not commit, upload, or describe it as synthetic.",
    }
    (out / "schema.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Prepared a bounded, local-only IPUMS pilot sample using public category domains.")
    print("Public OCC categories: 531; public IND categories: 266 (both include not-applicable).")
    print(f"Training rows: {args.train_rows}; held-out rows: {args.test_rows}")
    print(f"Data file: {(out / 'pilot_data.npz').resolve()}")
    print("Keep the output directory out of Git. This is NOT a DP data release.")


if __name__ == "__main__":
    main()
