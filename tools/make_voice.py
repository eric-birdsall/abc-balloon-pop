"""Generate the game's voice: every phrase it says, packed into one MP3 sprite.

Writes voice-<tag>.mp3 (all clips back to back, separated by silence) and
voice-<tag>.json ({key: [start_seconds, duration_seconds]}) next to abc-balloon-pop.html.
The keys must match the ones built in the game's `L` line helpers, and WORDS must
match the game's WORDS list.

Needs: kokoro-onnx, soundfile, numpy, Homebrew espeak-ng, lame, and the Kokoro
model files (kokoro.onnx, voices.bin) in the working directory.
Usage: python make_voice.py <output_dir> <kokoro_voice> <tag>
  e.g. make_voice.py .. af_heart f   and   make_voice.py .. am_puck m
"""
import json, subprocess, sys, tempfile, os
import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro, EspeakConfig

WORDS = """cat dog sun pig bus hat bee cow fox egg bed car
frog duck fish star moon cake bear tree boat ball lion milk
apple house horse train pizza mouse tiger snail robot whale
banana rabbit flower monkey rocket turtle spider carrot cookie guitar
rainbow dolphin giraffe penguin octopus balloon chicken cupcake
elephant dinosaur umbrella airplane mushroom kangaroo sandwich hedgehog""".split()
PRAISE = ["Great spelling!", "You did it!", "Super job!", "Awesome!", "Up, up and away!", "Wow, look at it go!"]

# Letter names as IPA, so "a" is never read as "uh" and each one sounds like the alphabet song.
LETTER_IPA = {
    "a": "ˈeɪ", "b": "bˈiː", "c": "sˈiː", "d": "dˈiː", "e": "ˈiː", "f": "ˈɛf", "g": "dʒˈiː",
    "h": "ˈeɪtʃ", "i": "ˈaɪ", "j": "dʒˈeɪ", "k": "kˈeɪ", "l": "ˈɛl", "m": "ˈɛm", "n": "ˈɛn",
    "o": "ˈoʊ", "p": "pˈiː", "q": "kjˈuː", "r": "ˈɑːɹ", "s": "ˈɛs", "t": "tˈiː", "u": "jˈuː",
    "v": "vˈiː", "w": "dˈʌbəljˌuː", "x": "ˈɛks", "y": "wˈaɪ", "z": "zˈiː",
}


def phrases():
    """{key: (text, is_phonemes)}"""
    p = {}
    for c, ipa in LETTER_IPA.items():
        p[f"l_{c}"] = (ipa, "letter")
        p[f"find_{c}"] = (f"fˈaɪnd ðə lˈɛɾɚ {ipa}.", True)
    for w in WORDS:
        p[f"w_{w}"] = (f"{w.capitalize()}.", False)
        p[f"ask_{w}"] = (f"Let's spell {w}.", False)
        p[f"spells_{w}"] = (f"spells {w}!", False)
    for i, t in enumerate(PRAISE):
        p[f"praise{i}"] = (t, False)
    for k, t in {
        "hello": "Let's spell with balloons!",
        "how": "Drag the balloons to the basket.",
        "levelup": "Wow! Now the words get longer!",
    }.items():
        p[k] = (t, False)
    return p


def trim(a, sr, thresh=0.01, pad=0.04):
    idx = np.where(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    s = max(0, idx[0] - int(pad * sr))
    e = min(len(a), idx[-1] + int(pad * sr))
    return a[s:e]


def segments(a, sr, min_gap=0.12, min_len=0.08, thresh=0.03):
    """[start, end) sample ranges of the spoken stretches in a, split at silences."""
    win = int(0.01 * sr)
    env = np.sqrt(np.convolve(a * a, np.ones(win) / win, mode="same"))
    on = env > thresh * env.max()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(np.int8), [0]])))
    segs = []
    for s, e in zip(edges[::2], edges[1::2]):
        if segs and s - segs[-1][1] < min_gap * sr:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    return [(s, e) for s, e in segs if e - s >= min_len * sr]


# Carrier phrases to cut a letter name out of: (phrase, shortest pause that counts as the
# break before the letter, how quiet that pause must be as a fraction of the peak).
# Tried in order until one leaves a clean pause for this voice.
_LOOK, _HERE = "lˈʊk æt ðə lˈɛɾɚ. {}.", "hˈɪɹ ɪz ðə lˈɛɾɚ. {}."
CARRIERS = [("ðə lˈɛɾɚ ɪz: {}.", 0.07, 0.03), (_LOOK, 0.03, 0.03), (_HERE, 0.03, 0.03),
            (_LOOK, 0.02, 0.06), (_HERE, 0.02, 0.06), ("wˈɑːt lˈɛɾɚ ɪz ðˈɪs? {}.", 0.03, 0.03)]


def letter_clip(k, ipa, voice):
    """A letter name on its own gives the model almost nothing to work with, and it puts a
    stray sound in front ("uh-see", "vee" for E). So say e.g. "The letter is: C." and cut the
    letter off the end at the pause. Returns (audio, sr, how)."""
    for n, (carrier, min_gap, thresh) in enumerate(CARRIERS):
        a, sr = k.create(carrier.format(ipa), voice=voice, speed=0.9, lang="en-us", is_phonemes=True)
        a = np.asarray(a, dtype=np.float32)
        segs = segments(a, sr, min_gap=min_gap, min_len=0.03, thresh=thresh)
        end = segs[-1][1]
        # the break before the letter is the longest pause that leaves a letter-sized tail
        gaps = [(s2 - e1, e1, s2) for (_, e1), (s2, _) in zip(segs, segs[1:])
                if 0.3 <= (end - s2) / sr <= 0.7 and e1 / sr >= 0.4]
        if gaps:
            gap, e1, s2 = max(gaps)
            pad = min(int(0.04 * sr), gap // 2)
            return a[s2 - pad:end + int(0.04 * sr)], sr, f"cut{n}"
    # no clean pause to cut at with this voice: fall back to the letter by itself
    a, sr = k.create(ipa + ".", voice=voice, speed=0.9, lang="en-us", is_phonemes=True)
    return trim(np.asarray(a, dtype=np.float32), sr), sr, "alone"


def main():
    out = sys.argv[1]
    voice, tag = sys.argv[2], sys.argv[3]
    k = Kokoro("kokoro.onnx", "voices.bin", espeak_config=EspeakConfig(
        lib_path="/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib",
        data_path="/opt/homebrew/opt/espeak-ng/share/espeak-ng-data"))
    ps = phrases()
    gap = 0.35
    chunks, index, pos, sr = [], {}, 0.0, 24000
    for i, (key, (text, phon)) in enumerate(ps.items()):
        if phon == "letter":
            a, sr, how = letter_clip(k, text, voice)
            print(f"  {key}: {how}, {len(a) / sr:.2f}s", flush=True)
        else:
            a, sr = k.create(text, voice=voice, speed=0.9, lang="en-us", is_phonemes=phon)
            a = trim(np.asarray(a, dtype=np.float32), sr)
        a = a / max(1e-6, np.abs(a).max()) * 0.89  # even loudness across clips
        index[key] = [round(pos + gap, 3), round(len(a) / sr, 3)]
        chunks += [np.zeros(int(gap * sr), np.float32), a]
        pos += gap + len(a) / sr
        if i % 25 == 0:
            print(f"{i}/{len(ps)} {key}", flush=True)
    chunks.append(np.zeros(int(gap * sr), np.float32))
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "voice.wav")
        sf.write(wav, np.concatenate(chunks), sr)
        subprocess.run(["lame", "--quiet", "-m", "m", "-b", "40", wav, os.path.join(out, f"voice-{tag}.mp3")], check=True)
    with open(os.path.join(out, f"voice-{tag}.json"), "w") as f:
        json.dump(index, f, separators=(",", ":"))
    print(f"done: {len(ps)} clips, {pos:.0f}s of audio")


if __name__ == "__main__":
    main()
