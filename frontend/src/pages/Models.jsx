import { Link } from "react-router-dom";
import { api } from "../api.js";
import { percent } from "../format.js";
import { useRequest } from "../hooks/useRequest.js";
import BarList from "../components/BarList.jsx";
import Status from "../components/Status.jsx";

const loadAll = async () => {
  const domains = await api.domains();
  return Promise.all(domains.map((d) => api.domain(d.id)));
};

function ConfusionMatrix({ classes, matrix }) {
  return (
    <div className="table-wrap">
      <table className="matrix">
        <thead>
          <tr>
            <th scope="col">Actual \ Predicted</th>
            {classes.map((c) => (
              <th key={c} scope="col">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={classes[i]}>
              <th scope="row">{classes[i]}</th>
              {row.map((count, j) => (
                <td key={classes[j]} className={i === j ? "diag" : undefined}>
                  {count}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Models() {
  const { data, error, loading } = useRequest(loadAll, []);

  return (
    <div className="stack-lg">
      <header className="page-head">
        <h1>Models</h1>
        <p className="muted">
          Evaluation on a stratified 25% held-out test split. Global importance is the drop in test accuracy when a
          feature is randomly shuffled (permutation importance, 10 repeats).
        </p>
      </header>

      <Status loading={loading} error={error}>
        {data?.map((domain) => (
          <section key={domain.id} className="card stack">
            <div className="section-head">
              <div>
                <span className="eyebrow">{domain.sector}</span>
                <h2>{domain.name}</h2>
              </div>
              <Link to={`/analyze/${domain.id}`} className="button">
                Open
              </Link>
            </div>

            <dl className="metrics">
              <div>
                <dt>Accuracy</dt>
                <dd>{percent(domain.metrics.accuracy, 1)}</dd>
              </div>
              <div>
                <dt>Macro F1</dt>
                <dd>{domain.metrics.macro_f1.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Train / test</dt>
                <dd>
                  {domain.metrics.n_train} / {domain.metrics.n_test}
                </dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{domain.model}</dd>
              </div>
            </dl>

            <div className="models-grid">
              <div className="stack">
                <h3>Global feature importance</h3>
                <BarList
                  label={`${domain.name} feature importance`}
                  items={domain.importance.map((i) => ({ key: i.key, label: i.label, value: i.mean }))}
                  format={(v) => `${(v * 100).toFixed(1)} pp`}
                />
              </div>
              <div className="stack">
                <h3>Confusion matrix</h3>
                <ConfusionMatrix classes={domain.classes} matrix={domain.metrics.confusion_matrix} />
              </div>
            </div>

            <p className="muted small">Data: {domain.data_source}</p>
          </section>
        ))}
      </Status>
    </div>
  );
}
