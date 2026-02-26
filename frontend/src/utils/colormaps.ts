/**
 * Colormap utilities for weight visualization.
 * Each colormap is defined as an array of RGB stops, interpolated linearly.
 */

type RGB = [number, number, number];

const COLORMAPS: Record<string, RGB[]> = {
  viridis: [
    [68, 1, 84], [72, 35, 116], [64, 67, 135], [52, 94, 141],
    [41, 120, 142], [32, 144, 140], [34, 167, 132], [68, 190, 112],
    [121, 209, 81], [189, 222, 38], [253, 231, 37],
  ],
  plasma: [
    [13, 8, 135], [75, 3, 161], [125, 3, 168], [168, 34, 150],
    [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40],
    [240, 249, 33],
  ],
  coolwarm: [
    [59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249],
    [221, 221, 221], [245, 196, 173], [244, 154, 123], [222, 96, 77],
    [180, 4, 38],
  ],
  magma: [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
    [181, 54, 122], [229, 89, 100], [251, 135, 97], [254, 194, 140],
    [252, 253, 191],
  ],
  inferno: [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
  ],
};

export type ColormapName = keyof typeof COLORMAPS;
export const COLORMAP_NAMES: ColormapName[] = Object.keys(COLORMAPS);

function interpolate(t: number, stops: RGB[]): RGB {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (stops.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, stops.length - 1);
  const frac = idx - lo;
  return [
    Math.round(stops[lo][0] + (stops[hi][0] - stops[lo][0]) * frac),
    Math.round(stops[lo][1] + (stops[hi][1] - stops[lo][1]) * frac),
    Math.round(stops[lo][2] + (stops[hi][2] - stops[lo][2]) * frac),
  ];
}

export function colormapRGB(value: number, name: ColormapName = 'viridis'): RGB {
  return interpolate(value, COLORMAPS[name] || COLORMAPS.viridis);
}

export function colormapCSS(value: number, name: ColormapName = 'viridis'): string {
  const [r, g, b] = colormapRGB(value, name);
  return `rgb(${r},${g},${b})`;
}
