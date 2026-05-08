"""Convert CelebA attribute annotations into prompt captions.

Usage:
    python make_celeba_captions.py \
        --attr_file list_attr_celeba.txt \
        --output data/celeba_captions.csv
"""

import argparse
import csv


ATTRIBUTE_PHRASES = {
    "5_o_Clock_Shadow": "with five o'clock shadow",
    "Arched_Eyebrows": "with arched eyebrows",
    "Bags_Under_Eyes": "with bags under the eyes",
    "Bald": "bald",
    "Bangs": "with bangs",
    "Big_Lips": "with full lips",
    "Big_Nose": "with a large nose",
    "Black_Hair": "with black hair",
    "Blond_Hair": "with blond hair",
    "Brown_Hair": "with brown hair",
    "Bushy_Eyebrows": "with bushy eyebrows",
    "Chubby": "with a round face",
    "Double_Chin": "with a double chin",
    "Eyeglasses": "wearing eyeglasses",
    "Goatee": "with a goatee",
    "Gray_Hair": "with gray hair",
    "Heavy_Makeup": "wearing heavy makeup",
    "High_Cheekbones": "with high cheekbones",
    "Mouth_Slightly_Open": "with mouth slightly open",
    "Mustache": "with a mustache",
    "Narrow_Eyes": "with narrow eyes",
    "Oval_Face": "with an oval face",
    "Pale_Skin": "with pale skin",
    "Pointy_Nose": "with a pointy nose",
    "Receding_Hairline": "with a receding hairline",
    "Rosy_Cheeks": "with rosy cheeks",
    "Sideburns": "with sideburns",
    "Smiling": "smiling",
    "Straight_Hair": "with straight hair",
    "Wavy_Hair": "with wavy hair",
    "Wearing_Earrings": "wearing earrings",
    "Wearing_Hat": "wearing a hat",
    "Wearing_Lipstick": "wearing lipstick",
    "Wearing_Necklace": "wearing a necklace",
    "Wearing_Necktie": "wearing a necktie",
    "Young": "young",
}

SKIP_ATTRIBUTES = {
    "Attractive",
    "Blurry",
    "Male",
    "No_Beard",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build prompt captions from CelebA attributes")
    parser.add_argument("--attr_file", required=True, help="Path to list_attr_celeba.txt or Kaggle CSV")
    parser.add_argument("--output", default="data/celeba_captions.csv", help="Output CSV path")
    parser.add_argument("--max_attributes", type=int, default=10, help="Maximum positive attributes per caption")
    parser.add_argument("--base_prompt", default="a portrait photo of a human face")
    return parser.parse_args()


def _read_txt_attr_file(path):
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]

    attributes = lines[1].split()
    for line in lines[2:]:
        parts = line.split()
        filename = parts[0]
        values = [int(value) for value in parts[1:]]
        yield filename, dict(zip(attributes, values))


def _read_csv_attr_file(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        filename_key = fieldnames[0]
        attributes = fieldnames[1:]
        for row in reader:
            filename = row[filename_key]
            yield filename, {attribute: int(row[attribute]) for attribute in attributes}


def read_attr_file(path):
    if path.lower().endswith(".csv"):
        return _read_csv_attr_file(path)
    return _read_txt_attr_file(path)


def attributes_to_prompt(attributes, base_prompt, max_attributes):
    phrases = []

    if attributes.get("Male") == 1:
        phrases.append("of a man")
    elif attributes.get("Male") == -1:
        phrases.append("of a woman")

    if attributes.get("No_Beard") == -1 and attributes.get("Goatee") != 1 and attributes.get("Mustache") != 1:
        phrases.append("with a beard")

    for attribute, value in attributes.items():
        if value != 1 or attribute in SKIP_ATTRIBUTES:
            continue
        phrase = ATTRIBUTE_PHRASES.get(attribute)
        if phrase is not None:
            phrases.append(phrase)

    if not phrases:
        return base_prompt
    return f"{base_prompt} {', '.join(phrases[:max_attributes])}"


def main():
    args = parse_args()
    rows = []
    for filename, attributes in read_attr_file(args.attr_file):
        rows.append(
            {
                "filename": filename,
                "prompt": attributes_to_prompt(attributes, args.base_prompt, args.max_attributes),
            }
        )

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} captions to {args.output}")


if __name__ == "__main__":
    main()
