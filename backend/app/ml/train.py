from app.ml.store import store


def main() -> None:
    store.load_all(retrain=True)
    for domain_id, trained in store.models.items():
        metrics = trained.metrics
        print(f"{domain_id:<8} accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
