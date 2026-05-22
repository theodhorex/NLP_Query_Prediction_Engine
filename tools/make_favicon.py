from PIL import Image
import sys
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
logo_path = os.path.join(ROOT, 'logo.png')
if not os.path.exists(logo_path):
    print('logo.png not found')
    sys.exit(1)

img = Image.open(logo_path).convert('RGBA')

# sample background color from corners (average)
w, h = img.size
corners = [img.getpixel((0,0)), img.getpixel((w-1,0)), img.getpixel((0,h-1)), img.getpixel((w-1,h-1))]
# use average RGB
avg_bg = tuple(sum(p[i] for p in corners)//4 for i in range(3))

# tolerance for background detection
tol = 12

px = img.load()
for y in range(h):
    for x in range(w):
        r,g,b,a = px[x,y]
        if abs(r-avg_bg[0]) <= tol and abs(g-avg_bg[1]) <= tol and abs(b-avg_bg[2]) <= tol:
            px[x,y] = (r,g,b,0)

# trim transparent border
bbox = img.getbbox()
if bbox:
    img = img.crop(bbox)

# make square by padding
size = max(img.width, img.height)
new_img = Image.new('RGBA', (size, size), (255,255,255,0))
new_img.paste(img, ((size - img.width)//2, (size - img.height)//2), img)

# resize to common favicon sizes and save
out_png = os.path.join(ROOT, 'favicon.png')
new_img_256 = new_img.resize((256,256), Image.LANCZOS)
new_img_256.save(out_png, format='PNG')
print('Saved', out_png)

# save .ico with multiple sizes
out_ico = os.path.join(ROOT, 'favicon.ico')
ico_sizes = [(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)]
icons = [new_img.resize(s, Image.LANCZOS) for s in ico_sizes]
icons[0].save(out_ico, format='ICO', sizes=ico_sizes)
print('Saved', out_ico)
