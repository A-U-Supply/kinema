"""kinēma CLI — invoked by `python -m kinema` and the a-u.supply worker."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from kinema import sources
from kinema.pipeline import run_pipeline

logger = logging.getLogger("kinema")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def _partition_inputs(paths: list[str]) -> tuple[list[Path], list[Path]]:
    """Sort --input files into (images, audio) by extension. Unknown → image."""
    images: list[Path] = []
    audio: list[Path] = []
    for raw in paths:
        p = Path(raw)
        ext = p.suffix.lower()
        if ext in _AUDIO_EXTS:
            audio.append(p)
        else:
            images.append(p)
    return images, audio


def _resolve_images(args: argparse.Namespace, workdir: Path, picked_images: list[Path]) -> list[Path]:
    if args.image_source == "picked":
        if not picked_images:
            raise SystemExit("--input must include at least one image when --image-source=picked")
        return picked_images

    media_types = ["image"]
    filters = json.loads(args.image_query) if args.image_query else None
    hits = sources.search_media(
        media_types=media_types, filters=filters, per_page=args.image_count
    )
    if not hits:
        raise SystemExit("image search returned no results")

    paths: list[Path] = []
    for hit in hits[: args.image_count]:
        dest = workdir / "images" / hit.filename
        sources.download(hit.download_url, dest)
        paths.append(dest)
    return paths


def _resolve_audio(args: argparse.Namespace, workdir: Path, picked_audio: list[Path]) -> tuple[Path, str | None]:
    """Returns (path, default_title) — title used if --title not set."""
    if args.audio_source == "upload":
        # Prefer an explicit --audio; otherwise fall back to a sniff'd --input file.
        if args.audio:
            return Path(args.audio), None
        if picked_audio:
            return picked_audio[0], None
        raise SystemExit("--audio-source=upload needs --audio or an audio file via --input")

    if args.audio_source == "random_release":
        url, title = sources.random_release_track()
        dest = workdir / "audio" / "track.mp3"
        sources.download(url, dest)
        return dest, title

    if args.audio_source == "pick_release":
        if not args.release_code or args.release_track is None:
            raise SystemExit("--release-code and --release-track required for pick_release")
        url, title = sources.find_track_stream_url(args.release_code, args.release_track)
        dest = workdir / "audio" / "track.mp3"
        sources.download(url, dest)
        return dest, title

    if args.audio_source == "pick_track":
        if not args.track or "|" not in args.track:
            raise SystemExit('--track required, format "RELEASE_CODE|TRACK_ID"')
        code, raw_id = args.track.split("|", 1)
        url, title = sources.track_url_by_id(code, raw_id)
        dest = workdir / "audio" / "track.mp3"
        sources.download(url, dest)
        return dest, title

    if args.audio_source == "search":
        filters = json.loads(args.audio_query) if args.audio_query else None
        hits = sources.search_media(media_types=["audio"], filters=filters, per_page=10)
        if not hits:
            raise SystemExit("audio search returned no results")
        hit = hits[0]
        dest = workdir / "audio" / hit.filename
        sources.download(hit.download_url, dest)
        return dest, None

    raise SystemExit(f"unknown --audio-source: {args.audio_source}")


def main() -> None:
    p = argparse.ArgumentParser("kinema")
    # Defaults below MUST match the manifest defaults — the a-u.supply worker
    # omits a flag whenever the param value equals the manifest default, so the
    # CLI must reach the same value via argparse defaults.
    p.add_argument("--input", nargs="*", help="image and/or audio paths (worker mounts /work/input/)")
    p.add_argument("--image-source", choices=["picked", "search", "random"], default="picked")
    p.add_argument("--image-query", help="JSON filter object for search index")
    p.add_argument("--image-count", type=int, default=120)
    p.add_argument("--audio", help="audio path (used when --audio-source=upload)")
    p.add_argument(
        "--audio-source",
        choices=["upload", "random_release", "pick_release", "search"],
        default="random_release",
    )
    p.add_argument("--audio-query", help="JSON filter object for search index")
    p.add_argument("--release-code")
    p.add_argument("--release-track", type=int)
    p.add_argument("--track", help='dropdown value: "RELEASE_CODE|TRACK_ID" — used when --audio-source=pick_track')
    p.add_argument("--recipe", default="hybrid-wash", help="recipe name (resolved against /app/recipes/) or path")
    p.add_argument("--aspect", choices=["16:9", "9:16", "1:1"], default="16:9")
    p.add_argument("--sec-per-image", type=float, default=1.5)
    p.add_argument(
        "--title-card",
        choices=["track_title", "song_title", "custom", "none"],
        default="track_title",
        help="track_title=use audio title; song_title=random from #song-titles; custom=use --title; none=no card",
    )
    p.add_argument("--title", help="text used when --title-card=custom")
    p.add_argument("--no-title", action="store_true", help="alias for --title-card=none")
    p.add_argument("--seed", type=int)
    p.add_argument("--output", "-o", required=True, type=Path)
    p.add_argument("--workdir", type=Path)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    workdir = args.workdir or args.output.parent / ".kinema-tmp"
    workdir.mkdir(parents=True, exist_ok=True)

    # Resolve recipe path: bare name → recipes/<name>.yaml
    recipe_arg = Path(args.recipe)
    if not recipe_arg.exists():
        for base in [Path("/app/recipes"), Path(__file__).parent.parent / "recipes"]:
            cand = base / f"{args.recipe}.yaml"
            if cand.exists():
                recipe_arg = cand
                break
        else:
            raise SystemExit(f"recipe not found: {args.recipe}")

    picked_images, picked_audio = _partition_inputs(args.input or [])
    image_paths = _resolve_images(args, workdir, picked_images)
    audio_path, default_title = _resolve_audio(args, workdir, picked_audio)

    mode = "none" if args.no_title else args.title_card
    if mode == "none":
        title_text = None
    elif mode == "custom":
        title_text = args.title or None
    elif mode == "song_title":
        title_text = sources.random_song_title()
    else:  # track_title
        title_text = default_title

    run_pipeline(
        recipe_path=recipe_arg,
        image_paths=image_paths,
        audio_path=audio_path,
        out_path=args.output,
        aspect=args.aspect,
        sec_per_image=args.sec_per_image,
        title_text=title_text,
        seed=args.seed,
        workdir=workdir,
    )
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
