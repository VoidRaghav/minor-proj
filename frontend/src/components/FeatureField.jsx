export default function FeatureField({ feature, value, onChange }) {
  const id = `field-${feature.key.replace(/\s+/g, "-")}`;
  const numeric = Number(value);
  const sliderValue = Number.isFinite(numeric) ? Math.min(Math.max(numeric, feature.min), feature.max) : feature.min;

  return (
    <div className="field">
      <div className="field-head">
        <label htmlFor={id}>{feature.label}</label>
        {feature.unit && <span className="field-unit">{feature.unit}</span>}
      </div>
      <div className="field-controls">
        <input
          type="range"
          aria-label={`${feature.label} slider`}
          min={feature.min}
          max={feature.max}
          step={feature.step}
          value={sliderValue}
          onChange={(event) => onChange(feature.key, event.target.value)}
        />
        <input
          id={id}
          type="number"
          inputMode="decimal"
          min={feature.min}
          max={feature.max}
          step="any"
          value={value}
          onChange={(event) => onChange(feature.key, event.target.value)}
        />
      </div>
      <div className="field-range">
        <span>{feature.min}</span>
        <span>{feature.max}</span>
      </div>
    </div>
  );
}
