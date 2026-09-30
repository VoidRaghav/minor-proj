export default function BarList({ items, max, format, highlight, label }) {
  const scale = max ?? Math.max(...items.map((item) => item.value), 1e-9);

  return (
    <div className="bar-list" role="table" aria-label={label}>
      {items.map((item) => (
        <div
          key={item.key}
          className={`bar-row${item.key === highlight ? " is-active" : ""}`}
          role="row"
          title={`${item.label}: ${format(item.value)}`}
        >
          <span className="bar-label" role="cell">
            {item.label}
          </span>
          <span className="bar-track" role="cell" aria-hidden="true">
            <span className="bar-fill" style={{ width: `${Math.max(item.value / scale, 0) * 100}%` }} />
          </span>
          <span className="bar-value" role="cell">
            {format(item.value)}
          </span>
        </div>
      ))}
    </div>
  );
}
