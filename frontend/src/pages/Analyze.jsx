import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import { percent } from "../format.js";
import { useRequest } from "../hooks/useRequest.js";
import BarList from "../components/BarList.jsx";
import ContributionChart from "../components/ContributionChart.jsx";
import FeatureField from "../components/FeatureField.jsx";
import Status from "../components/Status.jsx";

const defaultsOf = (domain) => Object.fromEntries(domain.features.map((f) => [f.key, String(f.default)]));

function validate(domain, values) {
  const parsed = {};
  for (const feature of domain.features) {
    const number = Number(values[feature.key]);
    if (values[feature.key] === "" || !Number.isFinite(number)) {
      return { error: `${feature.label} needs a number.` };
    }
    if (number < feature.min || number > feature.max) {
      return { error: `${feature.label} must be between ${feature.min} and ${feature.max}.` };
    }
    parsed[feature.key] = number;
  }
  return { parsed };
}

export default function Analyze() {
  const { domainId } = useParams();
  const { data: domain, error, loading } = useRequest(() => api.domain(domainId), [domainId]);
  const [values, setValues] = useState(null);
  const [truth, setTruth] = useState(null);
  const [analysis, setAnalysis] = useState({ result: null, error: null, pending: false });

  useEffect(() => {
    setValues(domain ? defaultsOf(domain) : null);
    setTruth(null);
    setAnalysis({ result: null, error: null, pending: false });
  }, [domain]);

  const checked = useMemo(() => (domain && values ? validate(domain, values) : null), [domain, values]);

  useEffect(() => {
    if (!checked?.parsed) return undefined;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setAnalysis((current) => ({ ...current, pending: true }));
      api
        .analyze(domainId, checked.parsed, controller.signal)
        .then((result) => setAnalysis({ result, error: null, pending: false }))
        .catch((err) => {
          if (err.name !== "AbortError") setAnalysis({ result: null, error: err, pending: false });
        });
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [checked, domainId]);

  const steps = useMemo(
    () => Object.fromEntries((domain?.features ?? []).map((f) => [f.key, f.step])),
    [domain]
  );

  const updateValue = (key, value) => {
    setTruth(null);
    setValues((current) => ({ ...current, [key]: value }));
  };

  const loadSample = async () => {
    try {
      const sample = await api.sample(domainId);
      setValues(Object.fromEntries(Object.entries(sample.values).map(([k, v]) => [k, String(v)])));
      setTruth(sample.label);
    } catch (err) {
      setAnalysis({ result: null, error: err, pending: false });
    }
  };

  const reset = () => {
    setValues(defaultsOf(domain));
    setTruth(null);
  };

  const result = analysis.result;
  const inputError = checked?.error;

  return (
    <Status loading={loading} error={error}>
      {domain && values && (
        <div className="stack-lg">
          <header className="page-head">
            <span className="eyebrow">{domain.sector}</span>
            <h1>{domain.name}</h1>
            <p className="muted">{domain.task}</p>
          </header>

          <div className="analyze-grid">
            <section className="card stack" aria-labelledby="inputs-title">
              <div className="section-head">
                <h2 id="inputs-title">Inputs</h2>
                <div className="actions">
                  <button type="button" className="button" onClick={loadSample}>
                    Load test sample
                  </button>
                  <button type="button" className="button button-ghost" onClick={reset}>
                    Reset
                  </button>
                </div>
              </div>
              <p className="muted small">Defaults are dataset medians. Results update as you change values.</p>
              <div className="fields">
                {domain.features.map((feature) => (
                  <FeatureField
                    key={feature.key}
                    feature={feature}
                    value={values[feature.key]}
                    onChange={updateValue}
                  />
                ))}
              </div>
            </section>

            <section className="stack" aria-live="polite">
              {inputError && <p className="status status-error">{inputError}</p>}
              {!inputError && analysis.error && <p className="status status-error">{analysis.error.message}</p>}
              {!result && !inputError && !analysis.error && <p className="status">Analysing…</p>}

              {result && (
                <div className={`stack${analysis.pending || inputError ? " is-stale" : ""}`}>
                  <div className="card result-head">
                    <div>
                      <span className="eyebrow">Prediction</span>
                      <p className="result-label">{result.prediction}</p>
                      {truth && (
                        <p className={`small ${truth === result.prediction ? "match" : "mismatch"}`}>
                          Actual label: {truth} ({truth === result.prediction ? "correct" : "incorrect"})
                        </p>
                      )}
                    </div>
                    <div className="result-confidence">
                      <span className="eyebrow">Confidence</span>
                      <p className="result-number">{percent(result.confidence)}</p>
                    </div>
                  </div>

                  <div className="card stack">
                    <h3>Class probabilities</h3>
                    <BarList
                      label="Class probabilities"
                      items={result.probabilities.map((p) => ({ key: p.label, label: p.label, value: p.probability }))}
                      max={1}
                      highlight={result.prediction}
                      format={(v) => percent(v, 1)}
                    />
                  </div>

                  <div className="card">
                    <ContributionChart result={result} steps={steps} />
                  </div>

                  <div className="card stack">
                    <h3>Summary</h3>
                    <ul className="summary">
                      {result.summary.map((line) => (
                        <li key={line}>{line}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </section>
          </div>

          <p className="muted small">
            Data: {domain.data_source}. <Link to="/models">Model details</Link>
          </p>
        </div>
      )}
    </Status>
  );
}
