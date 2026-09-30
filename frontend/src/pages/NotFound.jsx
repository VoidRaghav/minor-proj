import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="stack page-head">
      <h1>Page not found</h1>
      <p className="muted">
        This page does not exist. <Link to="/">Back to overview</Link>
      </p>
    </div>
  );
}
