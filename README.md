Python tool for generating pictures from geometric constructions

## Installation

This project now uses modern Python libraries. Install dependencies:

```bash
# Using conda (recommended)
conda activate laws  # or your environment name
pip install -r requirements.txt

# Or using pip with virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or install dependencies manually:

```bash
pip install numpy matplotlib
```

## Usage

### Interactive Viewer (preview.py)

View geometric constructions interactively with matplotlib:

```bash
python preview.py
```

**Controls:**

- **Arrow keys (Up/Down/Left/Right)**: Navigate between different constructions
- **Space**: Regenerate current construction
- **Escape**: Exit the viewer

### Testing

Run the simple test to verify matplotlib rendering:

```bash
python test_matplotlib.py
```

This will generate `test_output.png` showing various geometric shapes.

## Project Structure

### Converting GeoGebra files to simpler construction format:

- `ggb_expr.py`, `ggb_parsetab.py` - Parsing expressions in GeoGebra
- `read_ggb.py` - Main conversion file

### ggb-benchmark

Directory containing decoded dataset from GeoGebra http://dev.geogebra.org/trac/browser/trunk/geogebra/test/scripts/benchmark/prover/tests

### Decoded constructions to pictures

- `random_constr.py` - Main file for construction generation
- `preview.py` - Interactive viewer using matplotlib (keyboard navigation)

### Other files

- `geo_types.py` - Geometric types (Line, Point, Circle, ...) with matplotlib rendering
- `commands.py` - Implementation of construction commands (intersection, are_collinear, ...)
- `requirements.txt` - Python package dependencies

## Migration from Gtk/Cairo to Matplotlib

This codebase has been modernized from using Gtk 3.0 and Cairo to using Matplotlib:

**Benefits:**

- ✅ Modern, actively maintained libraries
- ✅ Cross-platform compatibility (works on macOS, Linux, Windows)
- ✅ Easy to install with pip
- ✅ Rich plotting capabilities
- ✅ Interactive and non-interactive backends
- ✅ Better documentation and community support

**Changes:**

- All geometric shapes (`Point`, `Line`, `Circle`, etc.) now use `matplotlib.axes.Axes` for rendering
- Interactive viewer uses matplotlib's event system instead of Gtk
- Simpler installation process (no system packages needed)
