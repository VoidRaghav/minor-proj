import { formatValue, percent, points } from "../format.js";

export default function ContributionChart({ result, steps }) {
  const { contributions, prediction, base_value: base, confidence } = result;
  const largest = Math.max(...contributions.map((c) => Math.abs(c.contribution)), 1e-9);

  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <h3>Why {prediction}?</h3>
        <p className="muted">
          Shapley contribution of each input to the probability of {prediction}, in percentage points.
          Reference {percent(base)} plus all contributions equals {percent(confidence)}.
        </p>
        <div className="legend">
          <span className="legend-item">
            <span className="swatch swatch-pos" />
            Supports {prediction}
          </span>
          <span className="legend-item">
            <span className="swatch swatch-neg" />
            Argues against
          </span>
        </div>
      </figcaption>
      <div className="contrib-list" role="table" aria-label="Feature contributions">
        {contributions.map((c) => {
          const width = (Math.abs(c.contribution) / largest) * 50;
          const positive = c.contribution >= 0;
          const valueText = formatValue(c.value, steps[c.key], c.unit);
          return (
            <div
              key={c.key}
              className="contrib-row"
              role="row"
              title={`${c.label} = ${valueText}: ${points(c.contribution)}`}
            >
              <div className="contrib-label" role="cell">
                <span>{c.label}</span>
                <span className="muted">{valueText}</span>
              </div>
              <div className="contrib-track" role="cell" aria-hidden="true">
                <span className="contrib-axis" />
                <span
                  className={`contrib-bar ${positive ? "pos" : "neg"}`}
                  style={positive ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
                />
              </div>
              <div className="contrib-value" role="cell">
                {points(c.contribution)}
              </div>
            </div>
          );
        })}
      </div>
    </figure>
  );
}
