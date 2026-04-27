
from PIL import Image, ImageDraw




images = [
    ("Baseline B\nFID 309.14", "experiments/expB_faces_img128_bs1_ch64_ep20/generated/grid_sample.png"),
    ("F100\nFID 181.60", "experiments/expF_faces_img128_bs1_ch64_ep100/generated/grid_sample.png"),
    ("F300 EMA\nFID ~62", "experiments/expF300_ema_faces_img128_bs1_ch64_ep300/generated_ema/grid_sample.png"),
    ("F600 EMA + Cosine LR\nFID 41.82", "experiments/expF600_ema_coslr_faces_img128_bs1_ch64_ep600/generated_epoch600/grid_sample.png"),
]

thumb_size = 256
label_h = 50
padding = 10

tiles = []

for label, path in images:
    img = Image.open(path).convert("RGB")
    img = img.resize((thumb_size, thumb_size))

    tile = Image.new("RGB", (thumb_size, thumb_size + label_h), "white")
    tile.paste(img, (0, label_h))

    draw = ImageDraw.Draw(tile)
    draw.multiline_text((8, 5), label, fill="black")

    tiles.append(tile)

# Create 2x2 grid
grid_w = 2 * thumb_size + padding
grid_h = 2 * (thumb_size + label_h) + padding

out = Image.new("RGB", (grid_w, grid_h), "white")

positions = [
    (0, 0),
    (thumb_size + padding, 0),
    (0, thumb_size + label_h + padding),
    (thumb_size + padding, thumb_size + label_h + padding),
]

for tile, pos in zip(tiles, positions):
    out.paste(tile, pos)

out.save("figures/progression_grid.png")
print("Saved figures/progression_grid.png")
