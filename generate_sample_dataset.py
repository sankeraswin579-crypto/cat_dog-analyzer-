from pathlib import Path
from PIL import Image, ImageDraw

base = Path('dataset')
for subset in ['train', 'test']:
    for label, color in [('cat', (200, 100, 200)), ('dog', (100, 200, 150))]:
        folder = base / subset / label
        folder.mkdir(parents=True, exist_ok=True)
        count = 3 if subset == 'train' else 2
        for i in range(1, count + 1):
            img = Image.new('RGB', (150, 150), color)
            draw = ImageDraw.Draw(img)
            draw.text((10, 60), f'{label} {i}', fill='white')
            img.save(folder / f'{label}_{i}.png')

print('Sample dataset created:')
for path in sorted(base.rglob('*.png')):
    print(path)
