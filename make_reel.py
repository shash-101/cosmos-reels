"""Make an Instagram reel for @data_tocosmos.

Claude writes the script from prompt.txt, then this renders a vertical
1080x1920 video of text slides over an animated starfield.

    python make_reel.py                       # Claude picks the topic
    python make_reel.py --topic "black holes"
    python make_reel.py --script output/<folder>/script.json   # re-render, no AI
"""
import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
W, H, FPS = 1080, 1920, 30
HANDLE = "@data_tocosmos"
MODEL = "claude-opus-5-5"

FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"
ACCENT = (120, 220, 255)
TEXT_WIDTH = 880

# Rules the renderer depends on. Creative direction lives in prompt.txt.
SYSTEM = """You write scripts for vertical Instagram reels made of text slides.
Return JSON only, matching the schema.
- hook: the first slide, max 10 words.
- slides: 4 to 7 slides. headline max 8 words, body max 28 words.
- caption: the Instagram caption, without hashtags.
- hashtags: list of hashtags, each starting with #."""

SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "hook": {"type": "string"},
        "slides": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["headline", "body"],
                "additionalProperties": False,
            },
        },
        "caption": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["topic", "hook", "slides", "caption", "hashtags"],
    "additionalProperties": False,
}


# ---------- AI script ----------

def write_script(prompt, topic):
    user = prompt
    if topic:
        user += f"\n\nTopic for this reel: {topic}"

    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": SCHEMA},
        },
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    if response.stop_reason == "refusal":
        sys.exit("Claude declined this prompt. Try a different topic or prompt.")
    if response.stop_reason == "max_tokens":
        sys.exit("Response was cut off. Try again.")

    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


# ---------- drawing helpers ----------

def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size=size)


def wrap(draw, text, fnt, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def make_nebula(rng):
    """Soft colored clouds on a dark navy background."""
    sw, sh = W // 4, H // 4
    yy, xx = np.mgrid[0:sh, 0:sw]
    img = np.zeros((sh, sw, 3)) + np.array([4, 5, 18])
    colors = [(90, 40, 160), (30, 70, 170), (150, 40, 120), (20, 110, 150)]
    for _ in range(6):
        cx, cy = rng.uniform(0, sw), rng.uniform(0, sh)
        sigma = rng.uniform(sw * 0.15, sw * 0.4)
        strength = rng.uniform(0.25, 0.55)
        color = np.array(colors[rng.integers(len(colors))])
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        img += blob[..., None] * color * strength
    small = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return small.resize((W, H), Image.BICUBIC)


class Starfield:
    def __init__(self, rng, count=450):
        self.nebula = make_nebula(rng)
        self.x = rng.uniform(0, W, count)
        self.y = rng.uniform(0, H, count)
        self.depth = rng.uniform(0.2, 1.0, count)
        self.phase = rng.uniform(0, 2 * math.pi, count)
        self.freq = rng.uniform(0.3, 1.2, count)

    def frame(self, t):
        img = self.nebula.copy()
        draw = ImageDraw.Draw(img)
        ys = (self.y - t * 25 * self.depth) % H  # closer stars drift faster
        glow = 0.65 + 0.35 * np.sin(2 * math.pi * self.freq * t + self.phase)
        for x, y, d, g in zip(self.x, ys, self.depth, glow):
            r = 0.6 + 2.0 * d * d
            v = int(255 * g * (0.4 + 0.6 * d))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(v, v, min(255, v + 20)))
        return img.convert("RGBA")


def text_card(headline, body="", big=False):
    """A transparent layer with the slide's text centered on a soft panel."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    f_head = font(FONT_BOLD, 104 if big else 84)
    f_body = font(FONT_REGULAR, 58)

    blocks = [(line, f_head, ACCENT) for line in wrap(draw, headline, f_head, TEXT_WIDTH)]
    body_lines = wrap(draw, body, f_body, TEXT_WIDTH) if body else []
    gap = 40 if body_lines else 0
    head_h = len(blocks) * int(f_head.size * 1.22)
    body_h = len(body_lines) * int(f_body.size * 1.35)
    blocks += [(line, f_body, (245, 245, 255)) for line in body_lines]

    total = head_h + gap + body_h
    top = (H - total) // 2 - 80
    pad = 60
    draw.rounded_rectangle(
        (W / 2 - TEXT_WIDTH / 2 - pad, top - pad, W / 2 + TEXT_WIDTH / 2 + pad, top + total + pad),
        radius=48, fill=(8, 10, 30, 150),
    )

    y = top
    for i, (line, fnt, color) in enumerate(blocks):
        is_head = fnt is f_head
        if not is_head and i > 0 and blocks[i - 1][1] is f_head:
            y += gap
        draw.text((W / 2, y), line, font=fnt, fill=color, anchor="ma",
                  stroke_width=2, stroke_fill=(0, 0, 0, 160))
        y += int(fnt.size * (1.22 if is_head else 1.35))
    return layer


def handle_layer():
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((W / 2, 120), HANDLE, font=font(FONT_BOLD, 44),
                               fill=(255, 255, 255, 210), anchor="ma")
    return layer


def ease_out(t):
    return 1 - (1 - t) ** 3


# ---------- video ----------

def build_slides(script):
    slides = [(text_card(script["hook"], big=True), 2.8)]
    for s in script["slides"][:8]:
        words = len(f"{s['headline']} {s['body']}".split())
        seconds = min(7.0, max(3.2, 1.2 + words / 3.0))
        slides.append((text_card(s["headline"], s["body"]), seconds))
    slides.append((text_card("Follow for more", HANDLE, big=True), 2.5))
    return slides


def render(script, out_dir):
    rng = np.random.default_rng()
    stars = Starfield(rng)
    handle = handle_layer()
    slides = build_slides(script)
    total = sum(sec for _, sec in slides)

    writer = imageio.get_writer(out_dir / "reel.mp4", fps=FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p", macro_block_size=8)
    t0 = 0.0
    for n, (card, seconds) in enumerate(slides):
        frames = int(seconds * FPS)
        for i in range(frames):
            local = i / FPS
            t = t0 + local
            fade_in = ease_out(min(1.0, local / 0.45))
            fade_out = min(1.0, (seconds - local) / 0.3)
            alpha = fade_in * fade_out

            frame = stars.frame(t)
            frame.alpha_composite(handle)
            faded = card.copy()
            faded.putalpha(card.getchannel("A").point(lambda a: int(a * alpha)))
            frame.alpha_composite(faded, dest=(0, int((1 - fade_in) * 50)))

            rgb = frame.convert("RGB")
            bar = ImageDraw.Draw(rgb)
            bar.rounded_rectangle((60, 60, W - 60, 66), radius=3, fill=(70, 70, 100))
            bar.rounded_rectangle((60, 60, 60 + (W - 120) * t / total, 66), radius=3, fill=ACCENT)
            writer.append_data(np.asarray(rgb))

            if n == 0 and i == frames // 2:
                rgb.save(out_dir / "cover.png")
        t0 += seconds
    writer.close()
    return total


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "reel"


def main():
    parser = argparse.ArgumentParser(description="Make a @data_tocosmos reel.")
    parser.add_argument("--topic", help="what the reel is about (optional)")
    parser.add_argument("--prompt", default=str(HERE / "prompt.txt"), help="prompt file to use")
    parser.add_argument("--script", help="render an existing script.json without calling AI")
    args = parser.parse_args()

    if args.script:
        script = json.loads(Path(args.script).read_text(encoding="utf-8"))
    else:
        prompt = Path(args.prompt).read_text(encoding="utf-8")
        print("Writing script with Claude...")
        script = write_script(prompt, args.topic)

    out_dir = HERE / "output" / f"{datetime.now():%Y%m%d-%H%M%S}-{slugify(script['topic'])}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "script.json").write_text(json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "caption.txt").write_text(
        script["caption"] + "\n\n" + " ".join(script["hashtags"]), encoding="utf-8")

    print(f"Rendering: {script['topic']}")
    seconds = render(script, out_dir)
    print(f"Done! {seconds:.0f}s reel saved to {out_dir}")


if __name__ == "__main__":
    main()
