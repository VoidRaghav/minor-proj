import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml.domains import DOMAINS
from app.ml.explainer import ShapleyExplainer


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def defaults(client, domain_id):
    detail = client.get(f"/api/domains/{domain_id}").json()
    return {f["key"]: f["default"] for f in detail["features"]}


def test_shapley_values_are_additive():
    rng = np.random.default_rng(0)
    weights = rng.normal(size=4)

    def predict(X):
        p = 1 / (1 + np.exp(-(X * weights).sum(axis=1)))
        return np.column_stack([1 - p, p])

    explainer = ShapleyExplainer(predict, rng.normal(size=(20, 4)))
    x = rng.normal(size=4)
    base, phi = explainer.explain(x)
    np.testing.assert_allclose(base + phi.sum(axis=0), predict(x[None])[0], atol=1e-9)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["models"] == sorted(DOMAINS)


def test_list_domains(client):
    ids = [d["id"] for d in client.get("/api/domains").json()]
    assert ids == list(DOMAINS)


def test_unknown_domain(client):
    assert client.get("/api/domains/unknown").status_code == 404


@pytest.mark.parametrize("domain_id", list(DOMAINS))
def test_analyze_explains_prediction(client, domain_id):
    response = client.post(f"/api/domains/{domain_id}/analyze", json={"values": defaults(client, domain_id)})
    assert response.status_code == 200
    body = response.json()
    total = body["base_value"] + sum(c["contribution"] for c in body["contributions"])
    assert total == pytest.approx(body["confidence"], abs=1e-6)
    assert body["prediction"] in DOMAINS[domain_id].classes


@pytest.mark.parametrize("domain_id", list(DOMAINS))
def test_sample_is_valid_input(client, domain_id):
    sample = client.get(f"/api/domains/{domain_id}/sample").json()
    response = client.post(f"/api/domains/{domain_id}/analyze", json={"values": sample["values"]})
    assert response.status_code == 200


def test_rejects_out_of_range(client):
    values = defaults(client, "crop")
    values["soil_ph"] = 14
    response = client.post("/api/domains/crop/analyze", json={"values": values})
    assert response.status_code == 422
    assert "Soil pH" in response.json()["detail"]


def test_rejects_missing_feature(client):
    values = defaults(client, "food")
    values.pop("ph")
    response = client.post("/api/domains/food/analyze", json={"values": values})
    assert response.status_code == 422
