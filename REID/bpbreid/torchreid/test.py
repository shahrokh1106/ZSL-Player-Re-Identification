def extract(cfg, model):
    extractor = FeatureExtractor(
        cfg,
        device='cpu',
        model = model
    )
    data = load_reid_dataset()

    out = []
    for item in data:
        result = extractor(item)

        parts = result[0]['parts']
        parts_reshaped = parts.view(11, -1)

        foreg= result[0]['foreg']

        r = torch.cat((parts_reshaped, foreg), dim=1)

        out.append(r.numpy())

    GetScores(out)
    print("Scores obtained")