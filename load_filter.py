# load_filter.py — Load and view saved filter design
# ---------------------------------------------------
import json
from pathlib import Path

from bom import print_bom
from plotting import plot_bode, mark_spec, show_and_save
from sallen_key_tf import compute_cascade_tf


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def list_saved_filters():
    root = Path("filter designs")
    filters = sorted(root.glob("*.json"))
    if not filters:
        print("No saved filter designs found.")
        return []
    for i, f in enumerate(filters, 1):
        print(f"[{i}] {f.name}")
    return filters


def main():
    filters = list_saved_filters()
    if not filters:
        return

    try:
        choice = int(input("Select a filter to load (index): ")) - 1
        assert 0 <= choice < len(filters)
    except (ValueError, AssertionError):
        print("Invalid selection.")
        return

    data = load_json(filters[choice])
    print(f"\nLoaded: {filters[choice].name}")
    print(f"Type: {data['type']}, Order: {data['order']}")
    print(f"Passband: {data['wp_hz']} Hz, Stopband: {data['ws_hz']} Hz")
    print(f"Ripple: {data['ripple_db']} dB, Attenuation: {data['atten_db']} dB")
    if data.get('verification'):
        v = data['verification']
        print(f"Verified ripple: {v['ripple_db_achieved']:.2f} dB, "
              f"attenuation: {v['atten_db_achieved']:.2f} dB")

    print_bom(data['sections'])

    show = input("Show plot on screen? [y/N]: ").strip().lower() in ("y", "yes")
    b, a = compute_cascade_tf(data['sections'])
    fig, axes = plot_bode(b, a, label=filters[choice].stem)
    mark_spec(axes, data['wp_hz'], data['ws_hz'], data['ripple_db'], data['atten_db'])
    show_and_save(fig, filters[choice].with_suffix('.png'), show=show)
    print(f"[OK] Plot saved to {filters[choice].with_suffix('.png')}")


if __name__ == "__main__":
    main()
