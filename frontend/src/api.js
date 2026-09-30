async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body;
}

export const api = {
  domains: () => request("/domains"),
  domain: (id) => request(`/domains/${id}`),
  sample: (id) => request(`/domains/${id}/sample`),
  analyze: (id, values, signal) =>
    request(`/domains/${id}/analyze`, {
      method: "POST",
      body: JSON.stringify({ values }),
      signal,
    }),
};
