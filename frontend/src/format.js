export const percent = (value, digits = 0) => `${(value * 100).toFixed(digits)}%`;

export const points = (value) => `${value >= 0 ? "+" : "−"}${Math.abs(value * 100).toFixed(1)} pp`;

export const decimalsFor = (step) => (step >= 1 ? 0 : Math.max(0, -Math.floor(Math.log10(step))));

export const formatValue = (value, step, unit) => {
  const text = Number(value).toLocaleString(undefined, {
    minimumFractionDigits: decimalsFor(step),
    maximumFractionDigits: decimalsFor(step),
  });
  return unit && !/^\d/.test(unit) ? `${text} ${unit}` : text;
};
