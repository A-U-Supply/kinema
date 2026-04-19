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


def _resolve_images(args: argparse.Namespace, workdir: Path) -> list[Path]:
    if args.image_source == "picked":
        if not args.input:
            raise SystemExit("--input required when --image-source=picked")
        return [Path(p) for p in args.input]

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


def _resolve_audio(args: argparse.Namespace, workdir: Path) -> tuple[Path, str | None]:
    """Returns (path, default_title) — title used if --title not set."""
    if args.audio_source == "upload":
        if not args.audio:
            raise SystemExit("--audio required when --audio-source=upload")
        return Path(args.audio), None

    if args.audio_source == "random_release":
        code, n, title = sources.random_release_track()
        url = sources.stream_track_url(code, n)
        dest = workdir / "audio" / f"{code}-{n}.audio"
        sources.download(url, dest)
        return dest, title

    if args.audio_source == "pick_release":
        if not args.release_code or args.release_track is None:
            raise SystemExit("--release-code and --release-track required for pick_release")
        url = sources.stream_track_url(args.release_code, args.release_track)
        dest = workdir / "audio" / f"{args.release_code}-{args.release_track}.audio"
        sources.download(url, dest)
        return dest, None

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
    p.add_argument("--input", nargs="*", help="image paths (used when --image-source=picked)")
    p.add_argument("--image-source", choices=["picked", "search", "random"], default="picked")
    p.add_argument("--image-query", help="JSON filter object for search index")
    p.add_argument("--image-count", type=int, default=120)
    p.add_argument("--audio", help="audio path (used when --audio-source=upload)")
    p.add_argument(
        "--audio-source",
        choices=["upload", "random_release", "pick_release", "search"],
        default="upload",
    )
    p.add_argument("--audio-query", help="JSON filter object for search index")
    p.add_argument("--release-code")
    p.add_argument("--release-track", type=int)
    p.add_argument("--recipe", required=True, help="recipe name (resolved against /app/recipes/) or path")
    p.add_argument("--aspect", choices=["16:9", "9:16", "1:1"], default="16:9")
    p.add_argument("--sec-per-image", type=float, default=1.5)
    p.add_argument("--title", help="title card text — overrides default from audio")
    p.add_argument("--no-title", action="store_true", help="skip title card even if audio supplies one")
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

    image_paths = _resolve_images(args, workdir)
    audio_path, default_title = _resolve_audio(args, workdir)

    title_text = None if args.no_title else (args.title or default_title)

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
