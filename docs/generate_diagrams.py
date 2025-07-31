#!/usr/bin/env python3
"""
Script to generate PNG diagrams from PlantUML files
Usage: uv run docs/generate_diagrams.py [--jar /path/to/plantuml.jar]
"""

import os
import sys
import subprocess
import glob
import argparse
import urllib.request
from pathlib import Path

def download_plantuml_jar(bin_dir: Path) -> Path:
    """
    Download PlantUML jar file to the bin directory
    Returns the path to the downloaded jar file
    """
    plantuml_url = "https://github.com/plantuml/plantuml/releases/latest/download/plantuml.jar"
    jar_path = bin_dir / "plantuml.jar"

    print(f"PlantUML jar not found. Downloading from: {plantuml_url}")
    print(f"Download location: {jar_path}")

    try:
        # Create bin directory if it doesn't exist
        bin_dir.mkdir(parents=True, exist_ok=True)

        # Download with progress indication
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\rDownloading: {percent}% ({downloaded:,}/{total_size:,} bytes)", end="", flush=True)

        urllib.request.urlretrieve(plantuml_url, jar_path, progress_hook)
        print("\n✓ PlantUML jar downloaded successfully")

        return jar_path

    except Exception as e:
        print(f"\n✗ Failed to download PlantUML jar: {e}")
        print("Please download manually from: https://plantuml.com/download")
        sys.exit(1)


def get_project_root() -> Path:
    """
    Find the project root directory by looking for pyproject.toml
    """
    current_dir = Path(__file__).parent.absolute()

    # Walk up the directory tree to find pyproject.toml
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # Fallback to current directory if pyproject.toml not found
    return current_dir


def parse_arguments():
    """
    Parse command line arguments
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
        """
    )

    parser.add_argument(
        "--jar", "-j",
        type=str,
        dest="jar_path",
        metavar="PATH",
        help="Path to PlantUML jar file (if not provided, will download to project bin/ directory)",
        required=False,
        default=None
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        dest="output_dir",
        metavar="DIR",
        help="Output directory for generated PNG files (default: docs/assets/)",
        required=False,
        default=None
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        dest="verbose",
        help="Enable verbose output",
        required=False,
        default=False
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        dest="force_download",
        help="Force download of PlantUML jar even if it already exists",
        required=False,
        default=False
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Get project root and set up paths
    project_root = get_project_root()
    bin_dir = project_root / "bin"

    # Set up output directory
    if args.output_dir:
        assets_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent.absolute()
        assets_dir = script_dir / "assets"

    # Create assets directory if it doesn't exist
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Determine PlantUML jar location
    if args.jar_path:
        plantuml_jar = Path(args.jar_path)
        if not plantuml_jar.is_file():
            print(f"Error: PlantUML jar file not found at: {plantuml_jar}")
            print("Please check the path or omit --jar to auto-download")
            sys.exit(1)
    else:
        # Check if jar already exists in bin directory
        plantuml_jar = bin_dir / "plantuml.jar"
        if not plantuml_jar.is_file() or args.force_download:
            if args.force_download and plantuml_jar.is_file():
                if args.verbose:
                    print("Force download requested, removing existing jar...")
                plantuml_jar.unlink()
            plantuml_jar = download_plantuml_jar(bin_dir)

    # Check if Java is available
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, check=True, text=True)
        if args.verbose:
            java_version = result.stderr.split('\n')[0] if result.stderr else "Unknown version"
            print(f"Java found: {java_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Java is not installed or not in PATH")
        print("Please install Java to run PlantUML")
        sys.exit(1)

    # Set up directories
    script_dir = Path(__file__).parent.absolute()
    docs_dir = script_dir

    # Create assets directory if it doesn't exist
    assets_dir.mkdir(exist_ok=True)

    print("Generating diagrams from PlantUML files...")
    if args.verbose:
        print(f"Project root: {project_root}")
        print(f"PlantUML jar: {plantuml_jar}")
        print(f"Docs directory: {docs_dir}")
    print(f"Output directory: {assets_dir}")
    print()

    # Find all .puml files in docs directory only (not subdirectories)
    puml_pattern = str(docs_dir / "*.puml")
    puml_files = glob.glob(puml_pattern)

    if not puml_files:
        print(f"No .puml files found in {docs_dir} directory")
        return

    print(f"Found {len(puml_files)} .puml file(s):")
    for f in puml_files:
        print(f"  - {os.path.basename(f)}")
    print()

    # Process each PlantUML file
    processed = 0
    total = len(puml_files)

    for puml_file in puml_files:
        if args.verbose:
            print(f"Processing: {os.path.basename(puml_file)}")
        else:
            print(f"Processing: {os.path.basename(puml_file)}")

        # Build PlantUML command
        cmd = [
            "java",
            "-DPLANTUML_LIMIT_SIZE=8192",
            "-jar", str(plantuml_jar),
            "-tpng",
            "-o", str(assets_dir),
            puml_file
        ]

        if args.verbose:
            print(f"  Command: {' '.join(cmd)}")

        try:
            # Run PlantUML
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                base_name = Path(puml_file).stem
                output_file = assets_dir / f"{base_name}.png"
                print(f"  ✓ Generated: {output_file}")
                processed += 1
            else:
                print(f"  ✗ Failed to generate: {os.path.basename(puml_file)}")
                print(f"    Exit code: {result.returncode}")
                if result.stderr:
                    print(f"    Error: {result.stderr.strip()}")
                if args.verbose and result.stdout:
                    print(f"    Output: {result.stdout.strip()}")

        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(puml_file)}: {e}")

        if not args.verbose:
            print()

    # Summary
    print("Summary:")
    print(f"  Processed: {processed}/{total} files")
    print(f"  Output directory: {assets_dir}")

    # List generated files
    if processed > 0:
        print()
        print("Generated files:")
        png_files = list(assets_dir.glob("*.png"))
        if png_files:
            for png_file in sorted(png_files):
                file_size = png_file.stat().st_size
                print(f"  - {png_file.name} ({file_size:,} bytes)")
        else:
            print("  No PNG files found in assets directory")

    print()
    print("Done!")

if __name__ == "__main__":
    main()
