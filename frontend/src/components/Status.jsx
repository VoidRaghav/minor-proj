export default function Status({ loading, error, children }) {
  if (loading) return <p className="status">Loading…</p>;
  if (error) return <p className="status status-error">{error.message}</p>;
  return children;
}
