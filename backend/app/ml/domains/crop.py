import numpy as np

from app.ml.spec import DomainSpec, Feature

FEATURES = (
    Feature("soil_moisture", "Soil moisture", "%", 0.5),
    Feature("nitrogen", "Nitrogen", "mg/kg", 1),
    Feature("phosphorus", "Phosphorus", "mg/kg", 1),
    Feature("potassium", "Potassium", "mg/kg", 1),
    Feature("soil_ph", "Soil pH", "", 0.1),
    Feature("temperature", "Air temperature", "°C", 0.5),
    Feature("humidity", "Relative humidity", "%", 0.5),
    Feature("chlorophyll", "Leaf chlorophyll", "SPAD", 0.5),
)

CLASSES = ("Healthy", "Nutrient Deficiency", "Water Stress", "Disease Risk")


def load(seed: int, n: int = 3000):
    rng = np.random.default_rng(seed)
    moisture = np.clip(rng.normal(32, 12, n), 5, 60)
    nitrogen = np.clip(rng.normal(80, 30, n), 10, 150)
    phosphorus = np.clip(rng.normal(40, 15, n), 5, 80)
    potassium = np.clip(rng.normal(160, 55, n), 20, 300)
    ph = np.clip(rng.normal(6.6, 0.8, n), 4.5, 9.0)
    temperature = np.clip(rng.normal(27, 6, n), 10, 42)
    humidity = np.clip(rng.normal(65, 16, n), 20, 100)
    chlorophyll = np.clip(18 + 0.22 * nitrogen + 0.04 * phosphorus + rng.normal(0, 4, n), 15, 60)

    deficiency = (
        (65 - nitrogen) / 18
        + (28 - phosphorus) / 10
        + (110 - potassium) / 40
        + (38 - chlorophyll) / 6
        + np.maximum(np.abs(ph - 6.5) - 0.8, 0) * 1.5
    )
    water = (26 - moisture) / 5 + (temperature - 29) / 4 + (48 - humidity) / 12
    disease = (humidity - 74) / 6 + (moisture - 40) / 7 + (5 - np.abs(temperature - 25)) / 4
    healthy = np.zeros(n)

    scores = np.column_stack([healthy, deficiency, water, disease])
    scores += rng.gumbel(0, 0.35, scores.shape)
    y = scores.argmax(axis=1)
    X = np.column_stack([moisture, nitrogen, phosphorus, potassium, ph, temperature, humidity, chlorophyll])
    return X, y


SPEC = DomainSpec(
    id="crop",
    name="Crop Health",
    sector="Agriculture",
    task="Crop health assessment from soil, weather and leaf sensor readings",
    description="Assesses field condition as healthy, nutrient deficient, water stressed or at risk of fungal disease.",
    data_source="Synthetic field sensor data generated from agronomic threshold rules (3,000 samples)",
    features=FEATURES,
    classes=CLASSES,
    loader=load,
)
