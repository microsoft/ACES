# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#!/usr/bin/env python3
"""
Script to generate PNG diagrams from PlantUML files.

Usage: uv run docs/generate_diagrams.py [--jar /path/to/plantuml.jar]

Run from the external/saber directory.
"""

import os
import subprocess
import sys
import urllib.request
import argparse
from pathlib import Path


def download_plantuml_jar(bin_dir: Path) -> Path:
    """
    Download PlantUML jar file to the bin directory.

    Args:
        bin_dir: Directory to store the downloaded jar file.

    Returns:
        Path to the downloaded jar file.
    """
    plantuml_url = "https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar"
    jar_path = bin_dir / "plantuml.jar"

    print(f"PlantUML jar not found. Downloading from: {plantuml_url}")
    print(f"Download location: {jar_path}")

    try:
        bin_dir.mkdir(parents=True, exist_ok=True)

        def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(
                    f"\rDownloading: {percent}% ({downloaded:,}/{total_size:,} bytes)",
                    end="",
                    flush=True,
                )

        urllib.request.urlretrieve(plantuml_url, jar_path, progress_hook)
        print("\n✓ PlantUML jar downloaded successfully")

        return jar_path

    except Exception as e:
        print(f"\n✗ Failed to download PlantUML jar: {e}")
        print("Please download manually from: https://plantuml.com/download")
        sys.exit(1)


def get_project_root() -> Path:
    """
    Find the project root directory by looking for pyproject.toml.

    Returns:
        Path to the project root directory.
    """
    current_dir = Path(__file__).parent.absolute()

    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    return current_dir


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="generate_diagrams",
        description="Generate PNG diagrams from PlantUML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run docs/generate_diagrams.py
  uv run docs/generate_diagrams.py --jar ~/tools/plantuml.jar
  uv run docs/generate_diagrams.py -j /usr/local/bin/plantuml.jar
        """,
    )

    parser.add_argument(
        "--jar",
        "-j",
        type=str,
        dest="jar_path",
        metavar="PATH",
        help="Path to PlantUML jar file (if not provided, will download to project bin/ directory)",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        dest="output_dir",
        metavar="DIR",
        help="Output directory for generated PNG files (default: docs/assets/diagrams/)",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        dest="verbose",
        help="Enable verbose output",
        required=False,
        default=False,
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        dest="force_download",
        help="Force download of PlantUML jar even if it already exists",
        required=False,
        default=False,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Get project root and set up paths
    project_root = get_project_root()
    bin_dir = project_root / "bin"

    # Docs directory is the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    docs_dir = script_dir

    # Set up output directory
    if args.output_dir:
        assets_dir = Path(args.output_dir)
    else:
        assets_dir = docs_dir / "assets" / "diagrams"

    assets_dir.mkdir(parents=True, exist_ok=True)

    # Determine PlantUML jar location
    if args.jar_path:
        plantuml_jar = Path(args.jar_path)
        if not plantuml_jar.is_file():
            print(f"Error: PlantUML jar file not found at: {plantuml_jar}")
            print("Please check the path or omit --jar to auto-download")
            sys.exit(1)
    else:
        plantuml_jar = bin_dir / "plantuml.jar"
        if not plantuml_jar.is_file() or args.force_download:
            if args.force_download and plantuml_jar.is_file():
                if args.verbose:
                    print("Force download requested, removing existing jar...")
                plantuml_jar.unlink()
            plantuml_jar = download_plantuml_jar(bin_dir)

    # Check if Java is available
    try:
        result = subprocess.run(
            ["java", "-version"], capture_output=True, check=True, text=True
        )
        if args.verbose:
            java_version = (
                result.stderr.split("\n")[0] if result.stderr else "Unknown version"
            )
            print(f"Java found: {java_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Java is not installed or not in PATH")
        print("Please install Java to run PlantUML")
        sys.exit(1)

    print("Generating diagrams from PlantUML files...")
    if args.verbose:
        print(f"Project root: {project_root}")
        print(f"PlantUML jar: {plantuml_jar}")
        print(f"Docs directory: {docs_dir}")
    print(f"Output directory: {assets_dir}")
    print()

    # Find all .puml files recursively in docs directory
    puml_files: list[str] = []
    for root, dirs, files in os.walk(docs_dir):
        # Skip reference and assets directories
        for skip_dir in ("reference", "assets"):
            if skip_dir in dirs:
                dirs.remove(skip_dir)
        for file in sorted(files):
            if file.endswith(".puml"):
                puml_files.append(os.path.join(root, file))

    if not puml_files:
        print(f"No .puml files found in {docs_dir} directory")
        return

    print(f"Found {len(puml_files)} .puml file(s):")
    for f in puml_files:
        rel_path = os.path.relpath(f, docs_dir)
        print(f"  - {rel_path}")
    print()

    # Process each PlantUML file
    processed = 0
    failed = 0
    total = len(puml_files)

    for puml_file in puml_files:
        rel_path = os.path.relpath(puml_file, docs_dir)

        print(f"Processing: {rel_path}")

        # Build PlantUML command — output all PNGs to the flat assets dir
        cmd = [
            "java",
            "-Djava.awt.headless=true",
            "-DPLANTUML_LIMIT_SIZE=8192",
            "-jar",
            str(plantuml_jar),
            "-tpng",
            "-o",
            str(assets_dir),
            puml_file,
        ]

        if args.verbose:
            print(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                base_name = Path(puml_file).stem
                output_file = assets_dir / f"{base_name}.png"
                print(f"  ✓ Generated: {output_file.name}")
                processed += 1
            else:
                print(f"  ✗ Failed to generate: {rel_path}")
                print(f"    Exit code: {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr.strip()}")
                if args.verbose and result.stdout:
                    print(f"    Output: {result.stdout.strip()}")
                failed += 1

        except Exception as e:
            print(f"  ✗ Error processing {rel_path}: {e}")
            failed += 1

        print()

    # Summary
    print("=" * 50)
    print("Summary:")
    print(f"  Processed: {processed}/{total} files")
    if failed > 0:
        print(f"  Failed:    {failed}/{total} files")
    print(f"  Output:    {assets_dir}")

    # List generated files
    if processed > 0:
        print()
        print("Generated files:")
        for file in sorted(assets_dir.iterdir()):
            if file.suffix == ".png":
                file_size = file.stat().st_size
                print(f"  - {file.name} ({file_size:,} bytes)")
    else:
        print("  No PNG files generated")

    print()
    print("Done!")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
