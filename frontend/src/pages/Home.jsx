import { Link } from "react-router-dom";
import { api } from "../api.js";
import { percent } from "../format.js";
import { useRequest } from "../hooks/useRequest.js";
import Status from "../components/Status.jsx";

const steps = [
  {
    title: "Enter measurements",
    text: "Adjust sensor, lab or clinical readings, or load a held-out test sample with a known label.",
  },
  {
    title: "Model predicts",
    text: "A random forest trained for that domain returns class probabilities.",
  },
  {
    title: "Prediction is explained",
    text: "Exact Shapley values split the probability into one contribution per input, relative to a reference set.",
  },
];

export default function Home() {
  const { data, error, loading } = useRequest(api.domains, []);

  return (
    <div className="stack-lg">
      <section className="intro">
        <h1>Explainable AI for Agriculture and Healthcare</h1>
        <p className="lead">
          Transparent predictions for disease diagnosis, crop health assessment and food quality analysis. Every
          result shows which inputs drove it and by how much.
        </p>
      </section>

      <Status loading={loading} error={error}>
        <section className="grid-3">
          {data?.map((domain) => (
            <Link key={domain.id} to={`/analyze/${domain.id}`} className="card card-link">
              <span className="eyebrow">{domain.sector}</span>
              <h2>{domain.name}</h2>
              <p className="muted">{domain.description}</p>
              <span className="card-foot">Test accuracy {percent(domain.accuracy, 1)}</span>
            </Link>
          ))}
        </section>
      </Status>

      <section className="stack">
        <h2>How it works</h2>
        <ol className="steps">
          {steps.map((step) => (
            <li key={step.title} className="card">
              <h3>{step.title}</h3>
              <p className="muted">{step.text}</p>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
