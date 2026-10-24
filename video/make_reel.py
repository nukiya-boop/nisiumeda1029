"""西梅田 禅園 Instagram リール（1080x1920 / 30秒）生成スクリプト

使い方: python3 video/make_reel.py [--preview]
  --preview  各カットの代表フレームを PNG で書き出すだけ（動画は作らない）
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "images"
OUT = ROOT / "video" / "zenen_kanpai_reel.mp4"

W, H = 1080, 1920
FPS = 30
DUR = 30.0
NFRAMES = int(FPS * DUR)

SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
SERIF_B = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
LATIN = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
GOLD = (214, 182, 122)
IVORY = (246, 240, 228)


def font(path, size):
    return ImageFont.truetype(path, size, index=0) if path.endswith(".ttc") else ImageFont.truetype(path, size)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = min(max(t, 0.0), 1.0)
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


# ---------------------------------------------------------------- 画像の読み込み

def load(name, max_side=2600):
    im = Image.open(IMG / name).convert("RGB")
    if max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    return im


def grade(arr):
    """暖色寄りのシネマ調グレーディング"""
    a = arr.astype(np.float32) / 255.0
    a = a ** 1.04
    a[..., 0] *= 1.04
    a[..., 2] *= 0.94
    lum = a.mean(axis=2, keepdims=True)
    a = lum + (a - lum) * 1.06
    return np.clip(a * 255, 0, 255).astype(np.uint8)


def make_vignette():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((x - W / 2) / (W * 0.75)) ** 2 + ((y - H / 2) / (H * 0.70)) ** 2)
    return np.clip(1.0 - 0.55 * np.clip(d - 0.35, 0, 1) ** 1.5, 0, 1)[..., None]


VIGNETTE = make_vignette()


def kenburns(src, t, cx0, cy0, z0, cx1, cy1, z1, size=(W, H)):
    """src から中心(cx,cy)（0-1）・ズームzで size のアスペクトに切り出す"""
    tw, th = size
    e = ease_in_out(t)
    cx = cx0 + (cx1 - cx0) * e
    cy = cy0 + (cy1 - cy0) * e
    z = z0 + (z1 - z0) * e
    base = min(src.width / tw, src.height / th)
    cw, ch = tw * base / z, th * base / z
    x0 = min(max(cx * src.width - cw / 2, 0), src.width - cw)
    y0 = min(max(cy * src.height - ch / 2, 0), src.height - ch)
    return src.transform(size, Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), Image.BICUBIC)


class Framed:
    """低解像度の横長写真用：ぼかし背景＋金の細枠で中央に額装"""

    def __init__(self, src, top=586):
        self.src, self.top = src, top
        bg = kenburns(src, 0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, (W // 4, H // 4))
        bg = bg.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BICUBIC)
        self.bg = Image.fromarray((np.asarray(bg).astype(np.float32) * 0.42).astype(np.uint8))
        self.fw, self.fh = 1000, 667  # 原寸に近いサイズで劣化を抑える

    def render(self, t, pan_from, pan_to):
        out = self.bg.copy()
        pic = kenburns(self.src, t, pan_from, 0.5, 1.08, pan_to, 0.5, 1.08, (self.fw, self.fh))
        x, y = (W - self.fw) // 2, self.top
        out.paste(pic, (x, y))
        d = ImageDraw.Draw(out)
        d.rectangle([x - 14, y - 14, x + self.fw + 13, y + self.fh + 13], outline=GOLD, width=2)
        return out


# ---------------------------------------------------------------- テロップ

def vtext_layer(columns, size, color=IVORY, spacing=1.35, col_gap=None, bold=False):
    """縦書き（右の列から読む）。columns は列ごとの文字列"""
    f = font(SERIF_B if bold else SERIF, size)
    col_gap = col_gap or int(size * spacing)
    hs = [f.getbbox(c, direction="ttb", features=["vert"])[3] for c in columns]
    lw = col_gap * len(columns) + size
    lh = max(hs) + size
    layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i, c in enumerate(columns):
        x = lw - size - i * col_gap - size // 2
        d.text((x, size // 4), c, font=f, fill=color + (255,), direction="ttb", features=["vert"])
    return layer


def htext_layer(text, size, color=IVORY, path=SERIF, tracking=0.0):
    f = font(path, size)
    widths = [f.getlength(c) for c in text]
    total = sum(widths) + tracking * size * (len(text) - 1)
    layer = Image.new("RGBA", (int(total) + size, int(size * 1.6)), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x = size / 2
    for c, w in zip(text, widths):
        d.text((x, size * 0.2), c, font=f, fill=color + (255,))
        x += w + tracking * size
    return layer


def shadowed(layer, radius=10, strength=0.85):
    a = np.asarray(layer.split()[3]).astype(np.float32)
    pad = radius * 3
    big = Image.new("L", (layer.width + pad * 2, layer.height + pad * 2), 0)
    big.paste(Image.fromarray(a.astype(np.uint8)), (pad, pad))
    sh = np.asarray(big.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) * strength
    out = Image.new("RGBA", big.size, (0, 0, 0, 0))
    out.putalpha(Image.fromarray(sh.astype(np.uint8)))
    out.alpha_composite(Image.new("RGBA", big.size, (0, 0, 0, 0)))
    out.alpha_composite(layer, (pad, pad))
    return out, pad


class Telop:
    """フェード＋ゆっくり浮き上がり＋上から下へのマスク表示（筆で書くような出方）"""

    def __init__(self, layer, pos, t_in, t_out, reveal="down", drift=18, rdur=0.9):
        self.layer, self.pad = shadowed(layer)
        self.pos, self.t_in, self.t_out = pos, t_in, t_out
        self.reveal, self.drift, self.rdur = reveal, drift, rdur

    def draw(self, frame, t):
        if t < self.t_in or t > self.t_out:
            return
        p = (t - self.t_in) / self.rdur
        q = (self.t_out - t) / 0.45
        alpha = ease(min(p, q, 1.0))
        if alpha <= 0:
            return
        lay = self.layer.copy()
        lw, lh = lay.size
        r = ease_out(p)
        if self.reveal in ("down", "right") and r < 1:
            n = lh if self.reveal == "down" else lw
            ramp = np.clip((r * (n + 160) - np.arange(n)) / 160.0, 0, 1)
            mask = ramp[:, None] if self.reveal == "down" else ramp[None, :]
            a = np.asarray(lay.split()[3]).astype(np.float32) * mask
            lay.putalpha(Image.fromarray(a.astype(np.uint8)))
        a = np.asarray(lay.split()[3]).astype(np.float32) * alpha
        lay.putalpha(Image.fromarray(a.astype(np.uint8)))
        dy = int(self.drift * (1 - ease_out(p)))
        x, y = self.pos
        if x == "c":  # 中央寄せ
            x = (W - lw) // 2 + self.pad
        frame.alpha_composite(lay, (int(x) - self.pad, int(y) - self.pad + dy))


def gold_rule(frame, t, t_in, t_out, x, y, length, vertical=False):
    if t < t_in or t > t_out:
        return
    p = ease_out((t - t_in) / 0.8) * ease(min(1, (t_out - t) / 0.45))
    d = ImageDraw.Draw(frame)
    L = int(length * p)
    if L <= 0:
        return
    if vertical:
        d.line([(x, y), (x, y + L)], fill=GOLD + (230,), width=2)
    else:
        d.line([(x - L // 2, y), (x + L // 2, y)], fill=GOLD + (230,), width=2)


# ---------------------------------------------------------------- トランジション

def tr_fade_blur(a, b, p):
    p = ease(p)
    blur = 14 * math.sin(math.pi * p)
    if blur > 0.5:
        a = a.filter(ImageFilter.GaussianBlur(blur))
        b = b.filter(ImageFilter.GaussianBlur(blur))
    return Image.blend(a, b, p)


def tr_shoji(a, b, p):
    """障子が左右に開くように前のカットが割れていく"""
    e = ease_in_out(p)
    zoom = 1.0 + 0.08 * (1 - e)
    bb = b.resize((int(W * zoom), int(H * zoom)), Image.BICUBIC)
    bb = bb.crop(((bb.width - W) // 2, (bb.height - H) // 2, (bb.width - W) // 2 + W, (bb.height - H) // 2 + H))
    out = bb.copy()
    off = int(W / 2 * e)
    out.paste(a.crop((0, 0, W // 2, H)), (-off, 0))
    out.paste(a.crop((W // 2, 0, W, H)), (W // 2 + off, 0))
    d = ImageDraw.Draw(out)
    if 0 < e < 1:
        for x in (W // 2 - off, W // 2 + off):
            d.line([(x, 0), (x, H)], fill=GOLD, width=3)
    return out


def tr_gold_wipe(a, b, p):
    """金のラインが斜めに走って切り替わる"""
    e = ease_in_out(p)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    diag = (x * 0.6 + y) / (W * 0.6 + H)
    edge = e * 1.2 - 0.1
    m = np.clip((edge - diag) / 0.04, 0, 1)[..., None]
    out = np.asarray(a).astype(np.float32) * (1 - m) + np.asarray(b).astype(np.float32) * m
    line = np.exp(-((diag - edge) / 0.006) ** 2)[..., None] * (1 - abs(2 * e - 1))
    out = out * (1 - line) + np.array(GOLD, np.float32) * line
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def tr_flash_zoom(a, b, p):
    """乾杯の瞬間：ズームしながら白く光って切り替わる"""
    if p < 0.5:
        e = ease(p * 2)
        z = 1 + 0.18 * e
        img = a.resize((int(W * z), int(H * z)), Image.BICUBIC)
    else:
        e = ease((p - 0.5) * 2)
        z = 1.18 - 0.18 * e
        img = b.resize((int(W * z), int(H * z)), Image.BICUBIC)
    img = img.crop(((img.width - W) // 2, (img.height - H) // 2, (img.width - W) // 2 + W, (img.height - H) // 2 + H))
    glow = math.exp(-((p - 0.5) / 0.14) ** 2)
    warm = Image.new("RGB", (W, H), (255, 244, 225))
    return Image.blend(img, warm, 0.75 * glow)


def tr_iris(a, b, p):
    """盃のような円が広がって次のカットへ"""
    e = ease_in_out(p)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt((x - W / 2) ** 2 + (y - H * 0.47) ** 2)
    R = e * math.hypot(W, H) * 0.6
    m = np.clip((R - r) / 30, 0, 1)[..., None]
    out = np.asarray(a).astype(np.float32) * (1 - m) + np.asarray(b).astype(np.float32) * m
    ring = np.exp(-((r - R) / 4) ** 2)[..., None] * (1 - e)
    out = out * (1 - ring) + np.array(GOLD, np.float32) * ring
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def tr_slide_up(a, b, p):
    """縦にすっと流れる（パララックス）"""
    e = ease_in_out(p)
    out = Image.new("RGB", (W, H))
    out.paste(a, (0, int(-H * 0.35 * e)))
    dark = Image.blend(out, Image.new("RGB", (W, H)), 0.5 * e)
    dark.paste(b, (0, int(H * (1 - e))))
    return dark


def light_leak(img, t, center_t, width=0.7, pos=(0.85, 0.15)):
    k = math.exp(-((t - center_t) / width) ** 2)
    if k < 0.02:
        return img
    y, x = np.mgrid[0:H:4, 0:W:4].astype(np.float32)
    d = np.sqrt((x - W * pos[0]) ** 2 + (y - H * pos[1]) ** 2) / (W * 0.9)
    g = np.exp(-d ** 2 * 2.2) * k * 0.55
    g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32)[..., None] / 255
    arr = np.asarray(img).astype(np.float32)
    col = np.array([255, 170, 90], np.float32)
    out = 255 - (255 - arr) * (1 - g * col / 255)  # スクリーン合成
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- 構成

def build():
    counter = Framed(load("1P3J1135.jpg"), top=690)
    private = Framed(load("1P3J1102.jpg"))
    table = Framed(load("1P3J1094.jpg"))
    champ = load("DSC00416修.jpg")
    pour = load("DSC00725修.jpg")
    beer = load("DSC02081.jpg")
    sake = load("DSC02071.jpg")
    couple = load("DSC00522修.jpg")

    # (開始, 終了, 描画関数(局所t 0-1))
    shots = [
        (0.0, 4.4, lambda t: counter.render(t, 0.62, 0.40)),
        (4.4, 7.9, lambda t: kenburns(champ, t, 0.5, 0.42, 1.0, 0.5, 0.36, 1.12)),
        (7.9, 11.6, lambda t: private.render(t, 0.40, 0.60)),
        (11.6, 15.0, lambda t: kenburns(pour, t, 0.5, 0.55, 1.15, 0.5, 0.48, 1.0)),
        (15.0, 18.6, lambda t: table.render(t, 0.35, 0.62)),
        (18.6, 22.0, lambda t: kenburns(beer, t, 0.44, 0.55, 1.0, 0.46, 0.50, 1.10)),
        (22.0, 25.4, lambda t: kenburns(sake, t, 0.46, 0.50, 1.12, 0.48, 0.55, 1.0)),
        (25.4, 30.0, lambda t: kenburns(couple, t, 0.5, 0.5, 1.0, 0.52, 0.48, 1.08)),
    ]
    TD = 0.7  # トランジション長
    transitions = [tr_fade_blur, tr_shoji, tr_gold_wipe, tr_iris, tr_flash_zoom, tr_slide_up, tr_fade_blur]

    telops = [
        # オープニング：縦書きのタイトル
        Telop(vtext_layer(["多様なお席で、", "様々な乾杯を。"], 66, col_gap=100), (W // 2 - 110, 110), 0.5, 4.1, rdur=1.4),
        Telop(htext_layer("Counter", 38, GOLD, LATIN, tracking=0.2), (60, 1395), 1.8, 4.1, reveal="right"),
        Telop(htext_layer("カウンター席", 42, tracking=0.25), (60, 1450), 2.0, 4.1, reveal="right"),
        Telop(htext_layer("西梅田 禅園", 40, GOLD, tracking=0.45), ("c", 1565), 1.6, 4.1, reveal="right"),
        # シャンパン
        Telop(vtext_layer(["泡のきらめきに、", "想いを重ねて。"], 64, col_gap=96), (110, 300), 5.0, 7.6),
        Telop(htext_layer("Champagne", 38, GOLD, LATIN, tracking=0.3), (110, 1520), 5.4, 7.6, reveal="right"),
        # 個室
        Telop(htext_layer("Private Room", 40, GOLD, LATIN, tracking=0.25), ("c", 380), 8.6, 11.3),
        Telop(htext_layer("個室", 56, tracking=0.6), ("c", 440), 8.8, 11.3),
        Telop(htext_layer("大切な方と、心ゆくまで。", 52, tracking=0.12), ("c", 1340), 9.2, 11.3, reveal="right"),
        # 注ぎ合う
        Telop(vtext_layer(["注ぎあう一献に、", "ご縁を結ぶ。"], 64, col_gap=96), (W - 300, 250), 12.2, 14.7),
        Telop(htext_layer("Sake", 38, GOLD, LATIN, tracking=0.3), (W - 260, 1520), 12.6, 14.7, reveal="right"),
        # テーブル席
        Telop(htext_layer("Table", 40, GOLD, LATIN, tracking=0.25), ("c", 380), 15.7, 18.3),
        Telop(htext_layer("テーブル席", 56, tracking=0.3), ("c", 440), 15.9, 18.3),
        Telop(htext_layer("仲間と囲む、賑やかな宴。", 52, tracking=0.12), ("c", 1340), 16.2, 18.3, reveal="right"),
        # ビール
        Telop(htext_layer("まずは一杯、", 60, tracking=0.12), (110, 200), 19.2, 21.7, reveal="right"),
        Telop(htext_layer("笑顔で乾杯。", 60, tracking=0.12), (300, 290), 19.6, 21.7, reveal="right"),
        Telop(htext_layer("Cheers!", 38, GOLD, LATIN, tracking=0.3), (110, 1520), 19.8, 21.7, reveal="right"),
        # 日本酒と旬の一皿
        Telop(htext_layer("旬の一皿に、", 64, tracking=0.12), (110, 260), 22.6, 25.1, reveal="right"),
        Telop(htext_layer("盃を交わして。", 64, tracking=0.12), (300, 360), 23.0, 25.1, reveal="right"),
        Telop(htext_layer("Seasonal Kaiseki", 38, GOLD, LATIN, tracking=0.2), (110, 1520), 23.2, 25.1, reveal="right"),
        # エンディング
        Telop(vtext_layer(["それぞれの席に、", "それぞれの乾杯を。"], 56, col_gap=88), (W - 280, 150), 25.9, 30.5, rdur=1.3),
        Telop(htext_layer("西梅田 禅園", 72, IVORY, SERIF_B, tracking=0.35), ("c", 1230), 27.0, 30.5, reveal="right"),
        Telop(htext_layer("ZEN-EN  NISHI-UMEDA", 32, GOLD, LATIN, tracking=0.35), ("c", 1355), 27.3, 30.5),
        Telop(htext_layer("大阪・西梅田 ハービスPLAZA B2F", 34, tracking=0.1), ("c", 1460), 27.7, 30.5),
        Telop(htext_layer("ご予約  06-6457-1002", 34, tracking=0.1), ("c", 1520), 27.9, 30.5),
    ]
    return shots, transitions, TD, telops


def compose(t, shots, transitions, TD, telops):
    # どのカットか
    idx = max(i for i, s in enumerate(shots) if s[0] <= t) if t >= 0 else 0
    s0, s1, fn = shots[idx]
    local = lambda i, tt: (tt - shots[i][0]) / (shots[i][1] - shots[i][0])
    # トランジション：境界の前後 TD/2
    frame = None
    for i in range(len(shots) - 1):
        b = shots[i][1]
        if b - TD / 2 <= t < b + TD / 2:
            p = (t - (b - TD / 2)) / TD
            a_img = shots[i][2](min(local(i, t), 1.05))
            b_img = shots[i + 1][2](max(local(i + 1, t), -0.05))
            frame = transitions[i](a_img, b_img, p)
            break
    if frame is None:
        frame = fn(local(idx, t))

    arr = grade(np.asarray(frame))
    arr = (arr.astype(np.float32) * VIGNETTE)
    # テロップを読みやすくする上下のグラデーション
    arr[:520] *= np.linspace(0.62, 1.0, 520)[:, None, None]
    arr[-560:] *= np.linspace(1.0, 0.55, 560)[:, None, None]
    # オープニングとエンディングは暗くしてタイトルを立たせる
    dim = 1.0 - 0.30 * ease((t - 26.6) / 0.8)
    arr *= dim
    # フェードイン／アウト
    arr *= ease(t / 0.8) * ease((DUR - t) / 0.5)
    # 粒子感（フィルムグレイン）
    rng = np.random.default_rng(int(t * FPS))
    arr += rng.normal(0, 3.2, (H // 2, W // 2, 1)).repeat(2, 0).repeat(2, 1)
    frame = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    frame = light_leak(frame, t, 4.4, pos=(0.9, 0.1))
    frame = light_leak(frame, t, 15.0, pos=(0.1, 0.85))
    frame = light_leak(frame, t, 25.4, pos=(0.85, 0.2))

    rgba = frame.convert("RGBA")
    for tl in telops:
        tl.draw(rgba, t)
    gold_rule(rgba, t, 1.8, 4.1, W // 2, 1545, 260)
    gold_rule(rgba, t, 27.2, 30.5, W // 2, 1435, 420)
    return rgba.convert("RGB")


def main():
    shots, transitions, TD, telops = build()
    if "--preview" in sys.argv:
        pdir = ROOT / "video" / "preview"
        pdir.mkdir(exist_ok=True)
        for t in [2.5, 4.4, 6.5, 7.9, 10.5, 11.6, 13.8, 17.2, 18.6, 20.8, 22.0, 24.2, 25.4, 29.0]:
            compose(t, shots, transitions, TD, telops).save(pdir / f"t{t:05.1f}.jpg", quality=85)
        return
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-shortest", "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k",
        str(OUT),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(NFRAMES):
        proc.stdin.write(compose(n / FPS, shots, transitions, TD, telops).tobytes())
        if n % 90 == 0:
            print(f"{n}/{NFRAMES}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("done:", OUT)


if __name__ == "__main__":
    main()
