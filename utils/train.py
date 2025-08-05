from config import Config  # Make sure config.py exists and defines Config

def train_model(model, train_data, val_data, epochs, dataset_name):
    num_workers = 2  # Parallel data loading (2-4 for Mac)
    train_loader = DataLoader(
        train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )

    if Config.DEVICE == "mps":
        print("⚠️ Warning: Mixed precision on MPS is experimental and may cause instability. Falling back to float32.")
        use_mixed_precision = False
    else:
        use_mixed_precision = Config.DEVICE in ['cuda']

    scaler = torch.amp.GradScaler(enabled=use_mixed_precision)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"{dataset_name} Epoch {epoch+1}/{epochs}",
            unit="batch",
        )
        for batch in progress_bar:
            inputs = {
                k: v.to(Config.DEVICE, non_blocking=(Config.DEVICE == "cuda"))
                for k, v in batch.items()
            }
            if Config.DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**inputs)
                    loss = outputs.loss
            else:
                outputs = model(**inputs)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({
                'loss': f"{total_loss/batch_count:.4f}",
                'device': Config.DEVICE
            })

            # Memory cleanup
            if Config.DEVICE == "mps":
                torch.mps.empty_cache()
            elif Config.DEVICE == "cuda":
                torch.cuda.empty_cache()

            # Validation (optional, you may want to implement this)
            # val_metrics = evaluate_model(model, val_data, dataset_name)
            # print(f"Epoch {epoch+1} - Val Accuracy: {val_metrics['accuracy']:.4f}")
